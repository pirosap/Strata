// The EP placement rules (docs/ep-llm2-design.md, Track A + Track B shared policy): header-only, no GPU, no
// model.  The ranked list goes in, the owner list comes out; the split is checked against hand-computed
// ratios.  This is the first thing the hardware phase runs (Handover section of the design doc).
#include <cstdio>
#include <vector>
#include <array>
#include <cstdint>

#include "strata/program/ep_rules.hpp"

using namespace strata::program::ep_rules;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // ---- peer_share_prior: cap1 / (cap0 + cap1), 0.5 when both are 0, negative caps read as 0
    CHECK(peer_share_prior(0, 0) == 0.5);
    CHECK(peer_share_prior(100, 100) == 0.5);
    CHECK(peer_share_prior(100, 0) == 0.0);
    CHECK(peer_share_prior(0, 100) == 1.0);
    CHECK(peer_share_prior(-5, 100) == 1.0);
    CHECK(peer_share_prior(100, -5) == 0.0);
    CHECK(peer_share_prior(300, 100) == 0.25);   // the peer holds a quarter of the pair's room
    CHECK(peer_share_prior(100, 300) == 0.75);

    // ---- assign_cards: empty input, empty answer
    {
        const std::vector<ExpertRef> none;
        const std::array<int64_t, 2> cap{10, 10};
        const auto o = assign_cards(none, cap);
        CHECK(o.empty());
    }

    // ---- no room on either card: everything stays host
    {
        std::vector<ExpertRef> ranked(2);
        ranked[0] = ExpertRef{0, 0, 8};
        ranked[1] = ExpertRef{0, 1, 8};
        const std::array<int64_t, 2> cap{0, 0};
        const auto o = assign_cards(ranked, cap);
        CHECK(o.size() == 2);
        CHECK(o[0] == kHost && o[1] == kHost);
    }

    // ---- room on the primary only: every expert the pair admits goes to card 0
    {
        std::vector<ExpertRef> ranked(3);
        for (int32_t e = 0; e < 3; ++e) ranked[(size_t) e] = ExpertRef{0, e, 4};
        const std::array<int64_t, 2> cap{16, 0};
        const auto o = assign_cards(ranked, cap);
        CHECK(o.size() == 3);
        CHECK(o[0] == 0 && o[1] == 0 && o[2] == 0);
    }

    // ---- equal room: the hot head alternates 0,1,0,1 (a tie in the used/capacity ratios goes to the primary)
    {
        std::vector<ExpertRef> ranked(4);
        for (int32_t e = 0; e < 4; ++e) ranked[(size_t) e] = ExpertRef{0, e, 1};
        const std::array<int64_t, 2> cap{10, 10};
        const auto o = assign_cards(ranked, cap);
        CHECK(o.size() == 4);
        CHECK(o[0] == 0 && o[1] == 1 && o[2] == 0 && o[3] == 1);
    }

    // ---- 2:1 room: the hot head splits 2 to the roomier card, 1 to the tighter, the 4th falls to the host
    // hand-walk (bytes=1, cap 2 vs 1): e0 ties at 0/0 -> card 0; e1: 1/2 vs 0/1 -> card 1; e2: 2/2 fits 0
    // only -> card 0; e3: 3/2 fits neither -> kHost
    {
        std::vector<ExpertRef> ranked(4);
        for (int32_t e = 0; e < 4; ++e) ranked[(size_t) e] = ExpertRef{0, e, 1};
        const std::array<int64_t, 2> cap{2, 1};
        const auto o = assign_cards(ranked, cap);
        CHECK(o.size() == 4);
        CHECK(o[0] == 0 && o[1] == 1 && o[2] == 0 && o[3] == kHost);
        const auto c = count_owners(ranked, o);
        CHECK(c.experts[0] == 2 && c.experts[1] == 1 && c.experts[2] == 1);
        CHECK(c.bytes[0] == 2 && c.bytes[1] == 1 && c.bytes[2] == 1);
    }

    // ---- an expert that fits on neither card (bytes > both caps) is host; a later, smaller one still fits
    {
        std::vector<ExpertRef> ranked(2);
        ranked[0] = ExpertRef{0, 0, 11};
        ranked[1] = ExpertRef{0, 1, 10};
        const std::array<int64_t, 2> cap{10, 10};
        const auto o = assign_cards(ranked, cap);
        CHECK(o.size() == 2);
        CHECK(o[0] == kHost);
        CHECK(o[1] == 0);   // card 0 is at 0/10 used, card 1 at 0/10: the tie goes to the primary
    }

    // ---- count_owners: the kHost sentinel and any other owner value count as host (index 2);
    // a short owner vector is truncated to the ranked size (a residency prefix, the rigs' startup log)
    {
        std::vector<ExpertRef> ranked(4);
        for (int32_t e = 0; e < 4; ++e) ranked[(size_t) e] = ExpertRef{0, e, 1};
        const std::vector<int32_t> owner{0, 1, 7, kHost, 3};   // longer than ranked: the extra rows are ignored
        const auto c = count_owners(ranked, owner);
        CHECK(c.experts[0] == 1 && c.experts[1] == 1 && c.experts[2] == 2);
        CHECK(c.bytes[0] == 1 && c.bytes[1] == 1 && c.bytes[2] == 2);
        const std::vector<int32_t> prefix{0, 1};
        const auto cp = count_owners(ranked, prefix);
        CHECK(cp.experts[0] == 1 && cp.experts[1] == 1 && cp.experts[2] == 0);
        CHECK(cp.bytes[0] == 1 && cp.bytes[1] == 1 && cp.bytes[2] == 0);
    }

    // ---- the spread property itself: with equal room the two cards end with equal used bytes (the hot head
    // is split, not serialised on the primary) - the exact claim of the design doc
    {
        std::vector<ExpertRef> ranked(6);
        for (int32_t e = 0; e < 6; ++e) ranked[(size_t) e] = ExpertRef{0, e, 1};
        const std::array<int64_t, 2> cap{4, 4};
        const auto o = assign_cards(ranked, cap);
        const auto c = count_owners(ranked, o);
        CHECK(c.bytes[0] == 3 && c.bytes[1] == 3);   // 6 single-byte experts, 3 each, alternating
        CHECK(c.experts[0] == 3 && c.experts[1] == 3 && c.experts[2] == 0);
    }

    if (failed == 0) std::printf("ep_rules_test: ok\n");
    return failed ? 1 : 0;
}
