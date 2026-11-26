# EP for a dual CMP 90HX rig ("llm2") — draft design

**Status: DRAFT.** Branch `ep/llm2-draft` on the pirosap/Strata fork, based on upstream tag `v0.2.22`
(engine 0.1.69). Nothing in this branch is built or measured yet. Every speed figure below is either an
upstream number or the fork owner's own measurement on the target rig, marked as such; the draft's own
numbers do not exist until the hardware phase at the end of this document runs.

Upstream rules apply to this branch: EP is opt-in everywhere, and with EP off every code path is intended
to stay identical to v0.2.22.

## Target hardware

One specific machine, deliberately not a general profile:

| Part | What it is |
|---|---|
| GPUs | 2× NVIDIA CMP 90HX (GA102, sm_86, 10 GiB each), mining cards unlocked by a patched open driver (compute + PCIe Gen2 + CUDA P2P all enabled and proven on the rig) |
| Links | Each card on its own PCIe Gen2 x16 slot from the CPU root complex, ~6.3 GB/s host-to-device per card, independent of the other card. CUDA P2P between the cards works, ~5.3 GB/s. No NVLink. |
| Host | i9-7960X (16 cores / 32 threads, AVX-512), 96 GiB DDR4-2400 (192 GiB planned), Ubuntu 24.04 |
| Power | Both cards capped at 210 W sustained; the second card thermally throttles near ~200 W under worst-case matmul until its pads are serviced |

Design centre: **20 GiB of combined VRAM as one expert cache, all remaining experts resident in host RAM,
and two slow-but-independent links.** No upstream profile targets this combination: upstream's multi-GPU
work assumes one fast card plus helpers, or Gen4/Gen5 links.

## Baselines to beat (owner-measured on this rig, upstream v0.2.20/v0.2.22)

- Qwen3.8-Flash-Next (SC117 IQ3_S, ~78 GiB) on **one** 90HX under stock Strata: decode **43.9–45.1 tok/s**,
  prefill **~825 tok/s**, context 262,144, KV int8. The second card contributes nothing to this seat today.
- Inkling-2-Mini on both cards, stock Strata layer split (SQ-15, experts all VRAM-resident): decode
  **49.8 / 47.6 / 45.0 tok/s** (short / 4K / 16K prompts), prefill ~960 tok/s, at the adopted context of
  65,536 (KV fp8). 131,072 does not fit usefully in layer split on 2×10 GiB.
- llama.cpp tensor-parallel on the same pair (dense 27B, for contrast): TP works on dense models; this
  project is about MoE, where the parallel unit is the expert, not the matrix slice.

## What upstream already has (recon summary, 2026-11-26)

EP is **not** missing from upstream — it is missing only from the primary Qwen engine:

- **Inkling (`strata-inkling`) has a working dual-GPU EP** in its main CUDA backend ("cuda-ep2",
  `engine/src/inkling2/cuda/ink_runner.cuh`): experts are sharded over both cards and computed in parallel,
  activations cross by P2P when `probe_p2p()` (a real round-trip copy with checksum, not just
  `cudaDeviceCanAccessPeer`) passes, and the loader falls back to pipeline layer split when it does not.
  A second, Maya-derived EP backend (`ep_dual_gpu.cu`) also ships in-tree but is unreachable from the
  default backend factory; this draft does not invest in it (it duplicates the live path and has no
  expert cache or prefetch).
- **The primary engine (`strata`, Flash-Next) has no EP.** Its second-card mechanism is `PeerExperts`
  (`engine/include/strata/core/peer_experts.hpp`): a disjoint, adaptive expert tier on the peer card,
  computed with the primary's own kernels (rows are bit-identical on either card), results returned through
  mapped pinned host rows. The verify plan already classifies every routed row as VRAM hit / PCIe share /
  CPU row / peer row — the dispatch skeleton an EP mode needs is already there.
- **Placement is already frequency-driven.** The expert profile's ranked order fills primary → peer →
  remote tiers; `ExpertCache::replace` and `PeerExperts::adapt` swap by live routing counts; prefetch
  (`RouterLookahead`, `ExpertSource::prefetch`) already exists. What is *not* frequency-driven is the
  **card choice**: Inkling's cuda-ep2 picks the peer's experts by a static per-layer stripe over expert ids
  (`want_peer`), and the primary engine fills the primary first and lets the peer take the leftovers.
- **P2P is treated with respect upstream, for a recorded reason:** on PHB/SYS topologies
  `cudaMemcpyPeerAsync` has silently moved nothing (#2666, see `p2p_topology_safe()` in
  `src/core/peer_experts.cpp`). Every inter-card transfer in this draft therefore stays behind a probe
  with a host-staging fallback. On this rig P2P is proven, but the probe stays.

So the project is two tracks, not a new engine:

- **Track A — Inkling:** keep cuda-ep2, replace its static stripe with a frequency-prior sharding over
  the combined VRAM (shared policy header below).
- **Track B — Flash-Next:** add an opt-in `--ep` mode that symmetrises `PeerExperts`: the sharding map is
  computed up front from the profile over both cards' budgets, instead of "what the primary does not hold".

## Shared policy: `engine/include/strata/program/ep_rules.hpp`

Header-only, GPU-free, unit-testable, in the style of `split_rules.hpp`. Both tracks call the same pure
function so the policy cannot drift between engines:

- `assign_cards(ranked, cap_bytes)` — walk the profile's ranked (hottest-first) experts; give each to the
  card with the lower used/capacity ratio among the cards it fits in (ties to the primary); what fits
  nowhere stays on the host tier. This spreads the hot head of the distribution over both cards in
  proportion to their room, which is what makes the two cards compute in parallel instead of one card
  owning every hot expert.
- `peer_share_prior(cap0, cap1)` — the capacity share, for split knobs.
- `count_owners(...)` — residency accounting for startup logs.

Ideas adopted from other forks (re-implemented or cited, no code copied; see the fork survey that
motivated this branch):

- Frequency-prior expert promotion/caching — kuraneko1/Strata's knapsack placement and Project Maya's
  `STRATA_GLM_PROMOTE_MIN` reach the same conclusion independently; upstream's own `route_prior_top10`
  is the same family. This draft's contribution is applying the prior to the *card choice*, across the
  combined 20 GiB.
- Pipelined windows (Hardin22/Strata-DualGPU, measured ×1.17–1.30 on layer split) — the fallback if EP
  does not pay on this rig: the same "make both cards work at once" goal inside layer split.
- Lane pricing (lighttransport/Strata, from Project Maya): price each expert's row as CPU-compute vs
  PCIe-transfer from measured costs. Not in this draft's code; it is the scheduling question the hardware
  phase must answer with numbers before anyone builds it (open question 3).
- Thread role separation (zapabob/Hakua): physical cores compute, SMT siblings do I/O and prefetch.
  A tuning note for the 16C/32T host, not engine code.
- Second-card role selection by measurement (MistVVK/XeStrata): EP vs peer tier vs layer split is a
  measured choice per model, not a fixed identity — the measurement plan below keeps all three arms.

## Track A — what is wired in this draft (Inkling)

All changes are additive and gated by one new knob; with it off, behaviour is v0.2.22's.

- `BackendOpts::ep_prior_placement` (`engine/src/inkling2/ink_backend.h`), set by `--ep-prior-placement`
  or `IK2_EP_PRIOR_PLACEMENT` (`ink_main.cpp`, next to the existing `--ep-split` / `IK2_EP_SPLIT`).
- The flag is plumbed through `InkCudaBackend::init` into `InkCudaModel::load_from_model_file` as a
  trailing defaulted parameter (`ink_cuda_backend.cu`, `ink_runner.cuh`).
- In the loader's tier assignment, when the flag is on **and** cuda-ep2 is active, the per-expert card
  preference comes from `ep_rules::assign_cards` over the ranked load order with both cards' budgets
  (`bud0`/`bud1`), instead of the `want_peer` expert-id stripe. The existing fill loop, its cross-fill
  safety net and the host tier are unchanged; a startup line prints the resulting residency counts.
- Device-pair selection already honours `--gpus 0,1` / `IK2_GPUS` in the main backend (the {0,1}
  hardcoding exists only in the unreachable `ep_dual_gpu.cu`); the backend logs the resolved pair when
  the flag is on.
- P2P: **unchanged.** cuda-ep2 already attempts P2P only after `probe_p2p()`'s verified round trip and
  falls back to layer split otherwise; `STRATA_CUDA_PEER_PINNED=1` / `STRATA_CUDA_PEER_P2P=1` remain the
  manual arms. A TODO in the mode-selection block records the Gen2 question the hardware phase must A/B:
  on this rig H2D (~6.3 GB/s) is *faster* than P2P (~5.3 GB/s), so the probe's latency comparison, not
  habit, should decide.

Not in this draft (deliberately): usage-following swaps for the cuda-ep2 peer tier (the primary engine's
`PeerExperts::adapt` is the model if measurements ask for it), and any change to `ep_dual_gpu.cu`.

## Track B — what is sketched in this draft (Flash-Next)

A skeleton, structurally wired but not yet a working mode:

- `--ep` flag (`generate.cpp`): `Options::ep_mode`, parsed beside `--peer-device`, listed in the usage
  text. It **requires** `--peer-device N` (the existing flag names the pair's second card) and inherits
  `--peer-device`'s mutual exclusion with `--layer-split` and `--expert-cache-device1..3`; asking for
  `--ep` without a peer device is a startup error. Default off.
- `engine/include/strata/core/ep_peer.hpp`: the EP types on top of `PeerExperts` —
  `EpShardingMap` ((layer, expert) → primary / peer / host), `build_ep_sharding_map()` (the pre-open
  plan: ranked profile + per-pair bytes + both budgets → `ep_rules::assign_cards`),
  `map_from_residency()` (the post-open truth, from `ExpertCache::slot_of` and `PeerExperts::has`), and
  `ep_launch` / `ep_finish` dispatch-combine stubs that forward to `PeerExperts::launch` / `finish`
  (the rows the peer holds are exactly the plan's `kind == 2` rows today). TODO comments mark the two
  real pieces of kernel-level work: making the verify plan treat peer-held experts as first-class EP
  rows rather than overflow rows, and an opt-in device-side scatter to replace the mapped host rows.
- At the existing peer-open site in `generate.cpp`, `--ep` builds the residency map after `peer.open()`
  and logs the split (primary / peer / host counts). This is the seam where the hardware phase swaps in
  the pre-open map: `PeerExperts::open` currently fills "what the primary does not hold" in ranked
  order, so making the fill follow `build_ep_sharding_map` is a contained change to the fill loop, left
  as a marked TODO because it changes which experts the primary cache itself admits.

Prefill under EP uses the existing `--peer-prefill-rows` path unchanged in this draft.

## Config surface (new in this branch)

| Where | Knob | Default | Meaning |
|---|---|---|---|
| strata-inkling | `--ep-prior-placement`, env `IK2_EP_PRIOR_PLACEMENT` | off | cuda-ep2 card choice by frequency prior over the combined VRAM instead of the expert-id stripe |
| strata | `--ep` (with `--peer-device N`) | off | EP mode skeleton on the PeerExperts pair |
| both | existing `STRATA_CUDA_PEER_P2P` / `STRATA_CUDA_PEER_PINNED`, `STRATA_PEER_NO_P2P` | upstream | P2P arms and kill switch; unchanged |

## Measurement plan (hardware phase, in order)

Same-day A/B against a stock v0.2.22 build on the same machine, per upstream's community-benchmark rules.
Greedy token parity comes before any speed number: an EP arm that changes the tokens is a bug, not a result.

1. **Baselines first.** Stock Inkling `--gpus 2` in both auto modes (layer split and cuda-ep2 as upstream
   selects them), and stock Flash-Next single-card + `--peer-device` arms. These numbers anchor everything.
2. **Track A A/B.** `--ep-prior-placement` on/off under cuda-ep2, matched protocol (the rig's standing
   one: prefill at 4K/16K prompts, decode after them, 512-token generations). Plus the P2P arms
   (`STRATA_CUDA_PEER_P2P=1` vs `STRATA_CUDA_PEER_PINNED=1`).
3. **Track B.** `--ep` skeleton parity check (greedy identical to `--peer-device` alone), then the
   fill-order change, then parity + speed again.
4. Report what was *not* run (DSpark/MTP interaction, long-context 262K behaviour, thermals at 210 W
   over 10 minutes) as not run, not as zero.

## Open questions only hardware can answer

1. Is P2P actually faster than mapped pinned host staging on this rig for row-sized transfers? The
   bandwidths say almost nothing (5.3 vs 6.3 GB/s); latency and sync count should decide, and if P2P does
   not win, the honest outcome is to not use it.
2. Does the #2666 silent-corruption mode exist on this topology (single socket, two root ports, modded
   cards, patched driver)? The startup probe must pass *and* long runs should carry checksum spot-checks
   until this is settled.
3. Three-way scheduling: both cards plus the CPU pool can compute experts at once. Which rows belong
   where is a lane-pricing question (Gen2 link vs 16 AVX-512 cores vs VRAM bandwidth); a previous
   engine on this rig (RustingStrata) measured its CPU path as the bottleneck, so this is not assumed.
4. Primary-card VRAM pressure: dense + 262K KV + its expert share in 10 GiB — how much residency does
   the frequency prior actually buy over fill-first, and where is the context-vs-experts trade-off knee?
5. Thermals: EP loads both cards at once, unlike layer split's turns. Sustained clocks at the 210 W cap,
   and card 2's ~200 W memory-side ceiling, may tax EP more than split.
6. Prefill: decode is row-wise and suits EP; prefill is large batches. Does `--peer-prefill-rows`
   suffice, or does EP need its own chunking?
7. Drafting: Inkling's DSpark never fired in layer-split mode on this rig, and Maya's EP assumes the
   drafter lives on one GPU. How DSpark (Inkling) and MTP (Flash-Next) behave under EP is unmeasured.

## Handover — for the agents who build and tune this on the rig

Read this section first; the rest of the document is the why.

**Build (llm2, sm_86):**

```sh
cmake -S engine -B build-ep -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-ep -j --target strata strata-inkling
```

(Confirm the exact configure line against `docs/SOURCE_INSTALL.md` and the rig's existing build scripts;
upstream defaults `CMAKE_CUDA_ARCHITECTURES` to 120 with a warning when unset — an unset build will
produce binaries that load and then fail every kernel launch on sm_86.)

**What is real and what is stub, file by file:**

| File | State |
|---|---|
| `docs/ep-llm2-design.md` | This document. |
| `engine/include/strata/program/ep_rules.hpp` | Real, complete, pure functions. The first thing to unit-test (it needs no GPU): feed it a ranked list and check the split against hand-computed ratios. |
| `engine/src/inkling2/ink_backend.h`, `ink_main.cpp`, `cuda/ink_cuda_backend.cu`, `cuda/ink_runner.cuh` | Real wiring for `--ep-prior-placement` (Track A). Unbuilt: expect compile fixes around the new `load_from_model_file` trailing parameter and the `ep_rules` include. |
| `engine/include/strata/core/ep_peer.hpp` | Types are real; `ep_launch`/`ep_finish` are forwarding stubs with TODOs, by design. |
| `engine/src/program/generate.cpp` | `--ep` flag, validation, and the residency-map log at the peer-open site. The fill-order change is a marked TODO, **not** done. |

**Expected compile-fix areas** (the draft was written without a CUDA toolchain; treat these as the
likely first errors, in order): the `ep_rules.hpp` include path from `ink_runner.cuh`; the trailing
parameter's call sites other than `ink_cuda_backend.cu` (search `load_from_model_file` — the SYCL/HIP
runners have their own copies and do not need the flag); `ep_peer.hpp`'s use of `ExpertCache::slot_of`;
printf format specifiers in the new log lines.

**Order of work:** measurement plan step 1 (stock baselines) *before* fixing anything beyond compile
errors — an EP number without a same-day stock number is not a result. Then Track A A/B, then Track B.
Keep every change opt-in; if an arm is not faster on this rig, the correct outcome is to record that
and leave the default path alone.
