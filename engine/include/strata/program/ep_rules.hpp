// include/strata/program/ep_rules.hpp - rules for expert-parallel (EP) placement that do not need a GPU to check.
//
// DRAFT (branch ep/llm2-draft, see docs/ep-llm2-design.md).  Header-only on purpose, in the same style as
// split_rules.hpp: the placement policy is shared by the engines that wire EP up (strata-inkling's cuda-ep2 and
// the primary engine's --ep skeleton on PeerExperts), while the execution code stays per-engine, as everywhere
// else in this codebase.  Nothing here is on a default path: both engines consult these rules only when their
// EP mode is explicitly enabled.
//
// The one idea: an EP pair should not decide residency card-by-card ("the primary fills first, the peer takes
// what is left", or a fixed stripe by expert id).  The routed experts of a token are computed by whichever card
// holds them, in parallel, so the hot experts belong spread over BOTH cards in proportion to their room: a
// token whose top-k sits on one card alone serialises on that card while the other idles.  The profile's
// ranked order (routing frequency over a trace, the same order the caches already fill by) is the prior.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace strata::program::ep_rules {

/// One expert of the sharding problem, in the profile's ranked order (hottest first).
struct ExpertRef {
    int32_t layer = 0;
    int32_t expert = 0;
    int64_t bytes = 0;   ///< blob bytes on a card (same on both: the peer runs the primary's kernels)
};

/// Owner values returned by assign_cards: a card index (0 or 1), or kHost when neither card takes the expert
/// (it stays in the RAM tier and is computed as a CPU row / host expert, as today).
constexpr int32_t kHost = -1;

/// The share of the pair's combined expert room that sits on card 1, as a prior for a split knob
/// (Inkling's --ep-split): cap1 / (cap0 + cap1), 0.5 when both are 0.
inline double peer_share_prior(int64_t cap0, int64_t cap1) {
    const double total = (double) (cap0 > 0 ? cap0 : 0) + (double) (cap1 > 0 ? cap1 : 0);
    return total > 0.0 ? (double) (cap1 > 0 ? cap1 : 0) / total : 0.5;
}

/// Assigns every expert of `ranked` (hottest first) to a card, walking the ranked order and giving each expert
/// to the card with the lower used/capacity ratio among the cards it still fits in (ties go to card 0, the
/// primary, which also carries the dense weights).  An expert that fits on neither card is kHost.  Returns one
/// owner per input position (same indexing as `ranked`).
///
/// Walking in ranked order with the ratio rule spreads the hottest experts over both cards roughly in
/// proportion to their room, instead of filling card 0 with the whole hot head of the distribution.  The rule
/// is deliberately simple (no knapsack search): the profile order already carries the frequency information,
/// and the run-time adaptive swaps (ExpertCache::replace, PeerExperts::adapt) correct what the prior gets
/// wrong for a given conversation.
inline std::vector<int32_t> assign_cards(const std::vector<ExpertRef>& ranked, const std::array<int64_t, 2>& cap_bytes) {
    std::vector<int32_t> owner(ranked.size(), kHost);
    int64_t used[2] = {0, 0};
    for (size_t i = 0; i < ranked.size(); ++i) {
        const int64_t b = ranked[i].bytes;
        bool fits[2];
        double ratio[2];
        for (int c = 0; c < 2; ++c) {
            fits[c] = cap_bytes[(size_t) c] > 0 && used[c] + b <= cap_bytes[(size_t) c];
            ratio[c] = cap_bytes[(size_t) c] > 0 ? (double) used[c] / (double) cap_bytes[(size_t) c] : 1.0;
        }
        int pick = -1;
        if (fits[0] && fits[1]) pick = ratio[1] < ratio[0] ? 1 : 0;
        else if (fits[0]) pick = 0;
        else if (fits[1]) pick = 1;
        if (pick >= 0) {
            owner[i] = pick;
            used[pick] += b;
        }
    }
    return owner;
}

/// How many of the assigned experts (and bytes) each owner holds; index 0/1 are the cards, index 2 is kHost.
struct Counts {
    int64_t experts[3] = {0, 0, 0};
    int64_t bytes[3] = {0, 0, 0};
};

inline Counts count_owners(const std::vector<ExpertRef>& ranked, const std::vector<int32_t>& owner) {
    Counts c;
    const size_t n = ranked.size() < owner.size() ? ranked.size() : owner.size();
    for (size_t i = 0; i < n; ++i) {
        const int o = owner[i] == 0 ? 0 : owner[i] == 1 ? 1 : 2;
        c.experts[o] += 1;
        c.bytes[o] += ranked[i].bytes;
    }
    return c;
}

}  // namespace strata::program::ep_rules
