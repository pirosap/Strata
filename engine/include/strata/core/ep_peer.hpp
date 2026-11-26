// include/strata/core/ep_peer.hpp - DRAFT: expert-parallel (EP) mode for the primary engine, on PeerExperts.
//
// Branch ep/llm2-draft, see docs/ep-llm2-design.md.  Nothing here is on a default path: the primary engine
// reaches this code only under --ep, which itself requires --peer-device.
//
// The idea, in one paragraph.  Today the peer tier is asymmetric: the primary fills its expert cache first
// and the peer takes the ranked pairs the primary does not hold (PeerExperts::open).  EP reads the same two
// caches as ONE sharded pool: a sharding map decided up front from the expert profile assigns every
// (layer, expert) to the primary, the peer, or the host tier, with the hot head of the routing distribution
// spread over both cards (strata::program::ep_rules::assign_cards) so a token's top-k rows are computed by
// both cards in parallel.  The execution underneath does not change in this draft: the peer still computes
// its rows with the primary's kernels (bit-identical rows), still returns them through the mapped host rows,
// and the verify plan still sees the peer's share as the rows with kind == 2.  What changes first is who
// decides residency; the kernel-level work is listed as TODOs below, in the order the design doc's
// measurement plan takes them.
#pragma once

#include "strata/core/peer_experts.hpp"
#include "strata/program/ep_rules.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The EP sharding map: owner of every (layer, expert) in an --ep run.
/// owner values are the ep_rules ones: 0 = primary card, 1 = peer card, ep_rules::kHost (-1) = host tier.
struct EpShardingMap {
    std::vector<int32_t> owner;   ///< [n_layers * n_expert]
    int64_t n_layers = 0;
    int64_t n_expert = 0;

    int32_t at(int64_t layer, int64_t expert) const {
        return owner[(size_t) (layer * n_expert + expert)];
    }
    int64_t resident_on(int card) const {
        int64_t n = 0;
        for (const int32_t o : owner)
            if (o == card) ++n;
        return n;
    }
    int64_t host_resident() const { return resident_on(strata::program::ep_rules::kHost); }
};

/// The pre-open plan: sharding from the profile's ranked pairs, the per-pair blob bytes
/// (`bytes[(layer * n_expert) + expert]`) and both cards' expert budgets in bytes, by the shared
/// frequency-prior rule.  Pure: no GPU is touched, so this is the piece a CPU test can pin down.
inline EpShardingMap build_ep_sharding_map(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                           const std::vector<int64_t>& bytes, int64_t n_layers, int64_t n_expert,
                                           const std::array<int64_t, 2>& cap_bytes) {
    EpShardingMap m;
    m.n_layers = n_layers;
    m.n_expert = n_expert;
    m.owner.assign((size_t) (n_layers * n_expert), strata::program::ep_rules::kHost);
    std::vector<strata::program::ep_rules::ExpertRef> refs;
    refs.reserve(ranked.size());
    for (const auto& pr : ranked) {
        const int64_t idx = (int64_t) pr.first * n_expert + pr.second;
        const int64_t b = idx >= 0 && idx < (int64_t) bytes.size() ? bytes[(size_t) idx] : 0;
        refs.push_back({pr.first, pr.second, b});
    }
    const std::vector<int32_t> owners = strata::program::ep_rules::assign_cards(refs, cap_bytes);
    for (size_t i = 0; i < ranked.size(); ++i) {
        const int64_t idx = (int64_t) ranked[i].first * n_expert + ranked[i].second;
        if (idx >= 0 && idx < (int64_t) m.owner.size()) m.owner[(size_t) idx] = owners[i];
    }
    return m;
}

/// The post-open truth: the map the two existing fills actually produced (primary cache residency from
/// ExpertCache::slot_of, peer residency from PeerExperts::has).  Until the fills follow
/// build_ep_sharding_map, this is the map an --ep run really has, and the one its startup log prints.
inline EpShardingMap map_from_residency(const ExpertCache& primary, const PeerExperts& peer, int64_t n_layers,
                                        int64_t n_expert) {
    EpShardingMap m;
    m.n_layers = n_layers;
    m.n_expert = n_expert;
    m.owner.assign((size_t) (n_layers * n_expert), strata::program::ep_rules::kHost);
    for (int64_t l = 0; l < n_layers; ++l) {
        for (int64_t e = 0; e < n_expert; ++e) {
            if (primary.slot_of(l, e) >= 0) m.owner[(size_t) (l * n_expert + e)] = 0;
            else if (peer.has(l, e)) m.owner[(size_t) (l * n_expert + e)] = 1;
        }
    }
    return m;
}

/// Starts the peer's share of one layer in --ep mode: the entries i with kind[i] == 2, exactly
/// PeerExperts::launch today.
/// TODO(ep-llm2, hardware phase): the verify plan currently marks a row kind == 2 only when the peer holds
/// the expert AND the primary does not - peer rows are overflow rows.  A real EP plan decides the row's
/// card from the sharding map alone (a peer-held expert is a first-class row, not a fallback), which is a
/// change in the plan code (src/core/verify.cpp), not here.  `map` is carried through the signature so
/// that change does not move this call again; it is not consulted yet.
inline bool ep_launch(PeerExperts& peer, const EpShardingMap& map, int64_t layer, const float* x,
                      const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind, std::string& err,
                      float* out = nullptr) {
    (void) map;
    return peer.launch(layer, x, ids, n_tok, k, kind, err, out);
}

/// Waits for the launched share and writes its rows into `out`, exactly PeerExperts::finish today.
/// TODO(ep-llm2, hardware phase): the rows come back through mapped pinned host memory.  The device-side
/// arm - the peer scattering straight into the primary's device rows, or the primary reading the peer's
/// device rows over P2P - is opt-in material behind the topology probe (p2p_topology_safe in
/// src/core/peer_experts.cpp) and is only worth building if it measures faster than the host rows on the
/// target rig; on Gen2 links the host path's bandwidth is the higher of the two.
inline bool ep_finish(PeerExperts& peer, float* out, std::string& err) { return peer.finish(out, err); }

}  // namespace strata::core
