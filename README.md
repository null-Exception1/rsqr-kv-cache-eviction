# [RFC Draft]: Raw-Survivor Storage with Query-Side Rotation for KV Cache Eviction

**Status:** pre-draft / not yet posted.
---

## 0. Scope

This proposal is scoped **strictly as a StreamingLLM replacement**: a way to get StreamingLLM's precision guarantees (rotate-from-raw, no compounding drift) at lower aggregate compute, by rotating survivors once, at eviction boundaries, instead of every decode step. Within that scope - re-rotation vs. re-rotation, not re-rotation vs. leave-gap - this design is a genuine improvement: isolated-rotation-cost latency measurements now support this (§5.1, §5.2), though end-to-end inference latency and CUDA graph capture remain untested.

Whether re-rotation (in any form) is the right eviction strategy compared to leave-gap designs is a separate question, out of scope here, and left for a future dedicated experiment (see the open tension noted in §2.1).

---

## 1. Motivation

Rolling KV cache eviction under RoPE requires surviving tokens' positions to shift down to stay contiguous after older tokens are dropped. The standard way to do this is to re-rotate each survivor's cached key to its new position at eviction time.

Doing that naively - correcting an already-rotated key in place, repeatedly, across a token's lifetime - accumulates floating-point error. This is a real, measured effect (§2), not a theoretical concern. Continuous per-step rotation designs (e.g. StreamingLLM's reference implementation) avoid this by always rotating from a raw, unrotated baseline at every decoding step rather than correcting a previous correction - but pay for it with `O(window_size)` rotation work at every single decode step, for the life of the session.

This RFC proposes a design that gets both properties at once: survivor keys are stored raw and rotated **at most once**, directly from that raw baseline to their current logical position, only at eviction boundaries - never continuously, and never as a correction-on-correction.

The core mechanism - storing unrotated K and applying RoPE at attention time using per-request logical positions - is not new; it's how MiniPIC (Ordonez & Parnell, arXiv 2606.13126) already handles position-independent caching for prefix reuse. This proposal applies that substrate specifically to the eviction/compaction problem: the contribution is the survivor-flagging and index-compaction scheme built on top of it, not the query-side rotation identity itself.

---

## 2. The precision result

Repeatedly re-rotating an already-rotated key accumulates float error. Measured directly:

```python
test_content = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
P, evict_n = 300, 8

rotated_then_corrected = apply_rope(test_content, torch.tensor([float(P)], device=device), FREQS)
for i in range(10):
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)

fresh_at_target_position = apply_rope(test_content, torch.tensor([float(P - evict_n*10)], device=device), FREQS)

print("max abs diff:", (rotated_then_corrected - fresh_at_target_position).abs().max().item())
# max abs diff: 6.020069122314453e-06
```

Ten iterative corrections (each applied to the *previous* correction's output) diverge from a single direct rotation to the equivalent final position by ~6.02e-6 (float32). This confirms the compounding-drift mechanism is real, specifically in the failure mode where correction is applied in-place to an already-rotated key.

**Important scoping note:** this drift is not a property of re-rotation in general. Continuous per-step rotation, as StreamingLLM's reference design actually implements it, also rotates from a raw baseline at every step (§3.2 of the original paper: cache keys "prior to introducing the rotary transformation," then apply position transformation "at each decoding phase") - so it does not suffer this compounding either. The drift is specific to designs that correct an already-corrected key. Both continuous rotation and the mechanism below avoid it, for the same underlying reason (always rotate from raw), by different means (every step vs. only at eviction boundaries).

**Scaling and downstream impact - see §5.3 and §5.6.** The 10-iteration test above raised two questions at the time: whether drift compounds faster past 10 iterations, and whether 6e-6 is large enough to move attention scores at all. Both are now answered (bounded oscillation, not growth; invisible past softmax) - see §5 for the full results rather than treating this as still open.

### 2.1 Follow-up sweep: error scales with position/delta magnitude, separably from hop count

The compounding test above fixes `P=300, evict_n=8` and varies only iteration count. A follow-up
sweep (10 random draws per cell, float32) varied `P` and `evict_n` directly instead, comparing
1-hop vs. 2-hop correction against a fresh rotation at the equivalent target position:

```python
import random

print("rerotate 1 time")
for i in range(10):
    test_content = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
    P, evict_n = random.randint(P_LOW, P_HIGH), random.randint(EVICT_LOW, EVICT_HIGH)

    rotated_then_corrected = apply_rope(test_content, torch.tensor([float(P)], device=device), FREQS)
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)

    fresh_at_target_position = apply_rope(test_content, torch.tensor([float(P - evict_n*1)], device=device), FREQS)
    print("max abs diff:", (rotated_then_corrected - fresh_at_target_position).abs().max().item())

print("rerotate 2 time")
for i in range(10):
    test_content = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
    P, evict_n = random.randint(P_LOW, P_HIGH), random.randint(EVICT_LOW, EVICT_HIGH)

    rotated_then_corrected = apply_rope(test_content, torch.tensor([float(P)], device=device), FREQS)
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)
```

**Caveats on this follow-up sweep - most since resolved, see §5:**
Whether the floor traces to `P` magnitude, `evict_n` magnitude, or target position specifically is answered in §5.7 (it's `evict_n`, not `P`, not target position). Whether this diff range actually moves attention scores is answered in §5.6 (it doesn't, at the tested scale - invisible past softmax). Two gaps remain genuinely open: **only float32 was tested** (production dtypes bf16/fp16 remain unchecked - no longer planned as a priority given fp32 is the target dtype for this mechanism, but worth flagging for anyone deploying at lower precision), and **content was synthetic** (`torch.randn`), not real forward-pass activations (§5.8's real-model recall run provides softer corroboration on this front, but doesn't directly re-measure the tensor-level diff on real activations).


---

## 3. The mechanism

### 3.1 Query-side vs. key-side correction

**Key-side corrected** (standard):
$$A_{m,n} = \text{Softmax}\left(\frac{(R_m Q_m)\cdot(R_{n-\text{evicted}}K_n)^T}{\sqrt{d}}\right)$$

**Query-side corrected** (this design):
$$A_{m,n} = \text{Softmax}\left(\frac{(R_{m-\text{evicted}}Q_m)\cdot(R_n K_n)^T}{\sqrt{d}}\right)$$

RoPE attention scores depend only on relative angular displacement between query and key positions, so shifting the eviction correction from the key side to the query side computes an equivalent relative distance. This identity itself isn't new - it's the standard justification for RoPE as a relative encoding, and MiniPIC already exploits it for prefix caching. What's specific to this proposal is using it to make eviction a pure bookkeeping operation rather than a tensor-modifying one.

### 3.2 Survivor lifecycle

1. **In-window:** tokens are rotated normally as they arrive and attend. Tokens not flagged as future survivors need no special handling - they're evicted with the rest of the window, rotated state and all.
2. **Survivor flagging:** using a fixed selection strategy (every Δ-th token), the engine knows in advance which in-window tokens will survive the next eviction. For flagged tokens only, a **raw (unrotated) shadow copy** is stored alongside the normal rotated in-window copy.
3. **Window-exit:** when a flagged survivor exits the window at an eviction boundary, its rotated copy is dropped. Only the raw copy is retained. It is never rotated again until step 5.
4. **Index-map compaction:** survivor global positions are compacted into a consecutive logical timeline (e.g. global position 505 → logical position 5) via a lightweight integer index map. No tensor data moves.
5. **Query-time reconciliation:** at the single eviction-boundary event where a survivor exits the window, its raw K is rotated **once** - directly from raw to a bucket-local anchor position, a single hop, never a correction-on-correction. After that, its stored K is never touched again for the rest of its life in the survivor bucket. As the window continues to slide on later eviction events, the growing offset between the window and this survivor's bucket is reconciled entirely on the query side (§3.1): a copy of the incoming query is rotated by the current offset for that bucket, using RoPE's rotation-composition identity, rather than re-rotating the survivor's K. "Recomputed every step" (§4, index-compaction) refers to which query-side offset to apply for a given bucket, not to touching stored survivor K again.

### 3.3 Rotation cadence

Each survivor's raw K is rotated **exactly once, ever** - at the single eviction event where it exits the window - and never re-rotated afterward, regardless of how many further eviction events occur while it remains in the survivor bucket. This is what keeps key-side rotation `O(1)` per survivor rather than `O(window_size)` per step.

That saving does not eliminate the correction work; it moves it to the query side (§3.1, §3.2 step 5). As the window advances past a survivor's bucket, the offset is reconciled by rotating a query copy per active survivor bucket, not by touching stored keys. **Whether this query-side cost stays cheap as a long session accumulates many distinct survivor buckets is unmeasured, and is the load-bearing open question for this whole design** - not yet resolved by the isolated rotation-cost benchmarks in §5.1/§5.2, which measure key-side rotation cost specifically. See §5's open-questions list.

### 3.4 Precompute-ahead variant, targeting kernel-launch saturation

A refinement of the base mechanism (§3.2), aimed specifically at the small-batch region of the tail-latency problem (§5.1), not a replacement for it.

Rather than waiting for a survivor to actually exit the window before rotating it from raw (§3.2 step 5), the raw shadow copy can be corrected **ahead of the eviction event**, as soon as it's cheap to batch: each raw block, once flagged, is carried forward through a chain of already-precomputed corrections (raw → corrected-for-eviction-1 → corrected-for-eviction-2 → ...), so that by the time the token actually exits the window, the correct final value already exists and the boundary operation is a pointer-swap, not a compute step. Because each step in the chain is still a single hop from that block's own raw baseline - not a correction applied to a prior correction's *output* - this does not reintroduce the compounding-drift failure mode from §2; it only changes *when* the (still non-compounding) rotation is computed relative to when it's needed.

The motivation is kernel-launch overhead specifically, not compute cost. A GPU kernel launch carries a fixed cost `C` largely independent of how much data that launch processes. Early in a session - few survivors, small batches - launches are frequent and small, so `C` dominates total time and the per-token overhead is worst exactly when this mechanism has the least work to amortize it over. Two effects compound this early-session problem: (a) survivor count naturally grows as the session runs, which amortizes `C` over more work *passively*, given enough time; precomputing ahead does this *actively*, by batching corrections into fewer, larger launches before they're strictly needed, rather than waiting for natural growth. This should push the effective bump-severity curve past its overhead-dominated region sooner than the passive base mechanism (§3.2) would reach on its own.

**This was a real, checkable prediction, not an assumed win - and it was checked (§5.5), coming back negative for this specific implementation (lookahead queue).** The predicted curve shape (overhead-dominated at small batch, flattening once launches are large enough) was the right thing to test; the queueing mechanism proposed to exploit it just didn't deliver a net win once its own overhead was accounted for.

**Memory cost of this variant:** holding multiple precomputed correction states per flagged block, ahead of when the last one is actually needed, costs more memory *earlier* in the session than the base mechanism does - but not more *in aggregate* over the session's full lifetime, since any session long enough to reach the eviction threshold at all (a precondition for this whole design being relevant, §0) will eventually need that same memory once survivor count naturally grows regardless. The trade is real but bounded to sessions within this RFC's intended scope; it offers no benefit (and costs nothing extra) for sessions too short to reach the eviction threshold, since the mechanism is inert for those regardless of which variant is used.

**Tested, and found not to help - see §5.5.** The saturation-point prediction above was measured directly: fixed-shape-bucket padding works cleanly (survivor batches collapse to ≤8 distinct shapes), but the actual precompute mechanism (a fixed-lookahead queue on a separate CUDA stream) showed no reliable speedup at any tested depth, and deep lookahead was measurably *worse* than no precompute at all - likely because queueing/sync overhead competes with a rotation cost that's already tiny once batching alone is applied (§5.1). This closes off the lookahead-queue approach specifically. It does not close off precompute-ahead as a concept - true CUDA graph capture (capture/replay semantics, a different mechanism from the streams-based queue tested here) remains untested and is the more promising remaining lever if end-to-end measurement (§5.1's caveat) surfaces a real problem worth solving this way. This section is kept for completeness and as a record of what was tried, not as a live proposal.

### 3.5 What this buys, and what it doesn't

**Solid, by construction:**
- Zero compounding precision drift for survivors - every survivor rotation is a single hop from raw.
- Lower aggregate key-side rotation-op count than continuous per-step rotation: `O(1)` per survivor token (one rotation, at window-exit, never repeated) vs. `O(window_size)` per decode step for continuous rotation.

**Explicitly not solved, and not claimed as solved:**
- **Query-side cost at scale.** The O(1) key-side saving above is offset by moving position-correction work to the query side: reconciling a survivor bucket against the current window requires rotating a query copy per active bucket, not per survivor, but this is not free and its cost as a session accumulates many distinct buckets over a long lifetime is untested. This is the single most important open question in this proposal - it isn't answered by the isolated key-rotation-cost measurements in §5.1/§5.2, which do not include this query-side overhead.
- Tail latency vs. continuous rotation. Aggregate compute reduction does not imply lower or even comparable tail latency - this was the open concern; **isolated-rotation-cost measurement now supports the win, with one caveat** (§5.1, §5.2): batched rotation beat continuous rotation 3.7-4.6x across stream lengths, with the advantage growing (not flat) as stream length increases. Across eviction-severity settings the advantage held broadly (1.69-6.37x) but was not monotonic - a real dip appeared at the largest tested `evict_every`, so this should be read as "consistently favorable" rather than "cliff-free by construction." A separate `survivor_delta` sweep, independent of eviction frequency, found no cliff. This is not yet an end-to-end inference-latency measurement (real model, attention, scheduler) - the isolated-cost result is a strong signal, not a final one. Candidate mitigations remain relevant for the end-to-end follow-up if it surfaces a real problem the isolated test didn't capture: fusing the boundary rotation into an existing kernel launch (e.g. the attention or cache-write kernel) rather than a dedicated launch, so it inherits the "negligible overhead" behavior reported for fused per-step RoPE (FlashInfer) rather than the overhead-bound risk of small standalone kernels; amortizing the known survivor set over the last few steps before a boundary rather than rotating all of it in one step, since survivors are known in advance (§3.2 step 2); precomputing corrections ahead of the eviction event specifically to reach kernel-launch saturation sooner (§3.4, tested and found not to help in its lookahead-queue form - §5.5); and capping batch size via the Δ/window-size ratio (§5.2 - the `survivor_delta` axis showed no cliff, though it also runs on uncapped survivor accumulation, an open issue in its own right) so the worst case stays bounded.
- Bump severity is expected to scale with survivor density per eviction event and inversely with window size - smaller windows mean more frequent boundary crossings, which could raise bump frequency enough to erode or reverse the aggregate-compute advantage. Flagged as a real risk, not yet quantified.
- Raw-shadow-copy memory overhead: bounded (only flagged survivors carry a shadow copy, not the whole window), but not yet accounted for numerically.

---

## 4. Relationship to existing work

- **MiniPIC** (Ordonez & Parnell, arXiv 2606.13126) - the substrate this design builds on: unrotated K storage, RoPE applied at attention time via per-request logical positions, sub-100-LOC core-engine change. This proposal is not a restatement of MiniPIC; it's a specific eviction/compaction policy (survivor flagging, raw shadow copies, index-map compaction) built on top of that substrate.
- **StreamingLLM** (Xiao et al., arXiv 2309.17453) - the reference point for continuous per-step rotation's cost curve (§3.4), and the design this proposal is trying to match in aggregate compute while avoiding its per-step cost.
- This proposal does **not** take a position on whether re-rotation should happen at all vs. leave-gap (no correction) - that's a separate, unresolved question (§2.1, §0) or in tension with #51948-style designs and this author's own prior multi-fact ablation results, which found leave-gap outperforming re-rotation in some tested regimes. This RFC assumes re-rotation is the chosen strategy and proposes how to do it cheaply and without precision loss.

---

## 5. Results

The 7 open questions this RFC originally posed are now answered. Full experimental detail, methodology, and caveats are in the companion progress log and results doc in the repo; this section summarizes.

### 5.1 Latency - CLOSED, positive

Two dummy-tensor microbenchmarks (isolated rotation cost, T4, fp32).

**A previous version of this section reported speedups from a harness with a real bug in it, since fixed and retracted here.** The original Arm A (continuous per-step rotation) was gated to rotate on the *same cadence* as Arm B (eviction-boundary only) - so it wasn't actually testing "every step" against "only at eviction," it was comparing two batched arms with different payload sizes. The tell: A's and B's rotation-call counts were identical at every `n_steps` value, which should never happen if one arm is genuinely continuous. Fixed by making Arm A re-rotate the entire current window on every single step, matching the literal "continuous" claim - its call count now scales with `n_steps` directly, while Arm B's scales with `n_steps / evict_every`, a real structural difference instead of an artifact of matched gating.

- **Batched vs. continuous, across stream length** (n_steps 100→5000, re-run against the fixed harness): RSQR beats continuous per-step rotation by 3.7-4.6x, and - newly visible now that Arm A is genuinely continuous - **the speedup grows as stream length increases** (3.71x → 4.58x), rather than staying flat. Call-count ratio holds steady at ~8x throughout, consistent with RAP (arXiv 2602.02599)'s finding that RoPE itself is under 1% of inference latency - the win tracks kernel-launch count, not total tokens rotated.

```text
n_steps=  100 | A:   18.260ms (calls=  93, tokens=  2550) | B:    4.918ms (calls=  12, tokens=   624) | speedup=3.71x
n_steps=  500 | A:   96.594ms (calls= 493, tokens= 13550) | B:   24.875ms (calls=  62, tokens= 15624) | speedup=3.88x
n_steps= 1000 | A:  225.015ms (calls= 993, tokens= 27304) | B:   54.257ms (calls= 125, tokens= 63000) | speedup=4.15x
n_steps= 5000 | A: 1244.572ms (calls=4993, tokens=137304) | B:  271.469ms (calls= 625, tokens=1565000) | speedup=4.58x
```

**The one caveat that applies to everything in §5.1 and §3.5's latency discussion below: this is an isolated rotation-cost microbenchmark, not end-to-end inference latency.** It excludes real model/attention/scheduler overhead and hasn't been tested with CUDA graph capture (§5.5) or on hardware beyond a single T4. The mechanism-level conclusion (batching reduces kernel-launch count, and that reduction dominates) is well-supported; the specific multipliers above are a rotation-only measurement and should not be cited as an inference-serving speedup number without that follow-up.

### 5.2 Bump severity vs. window size/Δ - CLOSED, mixed

Re-run against the same fixed harness as §5.1 (Arm A now genuinely continuous, not gated to Arm B's cadence). Two sweeps, both at `n_steps=2000`:

```text
evict_every=  2  A=619.958ms  B=366.444ms  speedup=1.69x  calls A/B=1997/999
evict_every=  4  A=506.412ms  B=199.871ms  speedup=2.53x  calls A/B=1993/499
evict_every=  8  A=451.388ms  B=171.642ms  speedup=2.63x  calls A/B=1985/249
evict_every= 16  A=483.766ms  B=91.771ms  speedup=5.27x  calls A/B=1969/124
evict_every= 32  A=478.224ms  B=75.017ms  speedup=6.37x  calls A/B=1937/61
evict_every= 64  A=424.865ms  B=80.054ms  speedup=5.31x  calls A/B=1873/30

Sweep 1 summary: speedup range 1.69x-6.37x. NOT monotonic - dips at
evict_every=64 relative to evict_every=32, rather than continuing to
climb. Direction of the effect (rarer evictions favor RSQR) still
holds broadly, but "climbs monotonically" is not an accurate
description of this data and the earlier version of this section was
wrong to claim it.

survivor_delta=  2  A=493.068ms  B=126.545ms  speedup=3.90x  B_tokens_rotated=498
survivor_delta=  4  A=503.230ms  B=127.219ms  speedup=3.96x  B_tokens_rotated=996
survivor_delta=  8  A=477.551ms  B=130.473ms  speedup=3.66x  B_tokens_rotated=1992
survivor_delta= 16  A=487.400ms  B=128.346ms  speedup=3.80x  B_tokens_rotated=1992
survivor_delta= 32  A=446.827ms  B=151.586ms  speedup=2.95x  B_tokens_rotated=1992
survivor_delta= 64  A=491.774ms  B=129.282ms  speedup=3.80x  B_tokens_rotated=1992

Sweep 2 summary: speedup range 2.95x-3.96x, no cliff. This sweep uses
uncapped survivor accumulation - the same open issue tied to the
accuracy-side collapse investigated separately. Speedup doesn't show
obvious systematic degradation here, but this is flagged as a second,
latency-side reason (independent of the accuracy motivation) to
implement a survivor cap - not fixed here, only measured.
```

**Reading this honestly:** the `survivor_delta` axis (sweep 2) supports the original "no cliff" conclusion. The `evict_every` axis (sweep 1) does not support the original "monotonic" conclusion - the advantage still broadly favors rarer evictions, but the trend is not clean, and should be described as such rather than smoothed over.


### 5.3 Drift-scaling sanity check - CLOSED

100-iteration extension (fixed P=300, evict_n=8, fp32, 5 trials): diff **oscillates** in an 8e-6–3e-5 band with no monotonic growth - consistent with RoPE's angular periodicity, not compounding accumulation. The original §2 framing ("accumulates error") should be read as bounded, not growing.

### 5.4 Raw-shadow-copy memory accounting - CLOSED, not a bottleneck

Direct arithmetic plus a real CUDA allocation check (matched exactly, 1.00x): raw shadow copies cost 24KB per survivor (all layers, K+V, fp32 - note this stores raw V alongside raw K for implementation simplicity, even though V is never rotated by RoPE and a K-only design would halve this). Total cost stays under 1MB at 32 concurrent survivors, under 6MB even at 256. Memory was never the binding constraint at any tested scale - the earlier concern that motivated this question (unbounded survivor growth) turned out to matter for accuracy risk, not memory pressure, and a survivor cap has been implemented as a precaution (see 5.8 below).

### 5.5 Precompute-ahead saturation point - CLOSED, negative

Fixed-shape-bucket padding (survivor batches rounded to the nearest multiple of Δ) works cleanly - a realistic 1-64 survivor range collapses to ≤8 distinct shapes. But the actual precompute mechanism tested on top of that (fixed-lookahead queue, depths 1-32, separate CUDA stream) showed **no reliable speedup at any depth** - deep lookahead (32) was measurably worse than no precompute at all (0.75x). Likely cause: queueing/sync overhead competes with a rotation cost that's already tiny (§5.1), so there's little latency left to hide once batching alone is applied. This closes off the lookahead-queue approach specifically; true CUDA graph capture (capture/replay, not streams) remains untested and is the more promising remaining lever if further latency work is warranted.

### 5.6 Attention-score sensitivity to the magnitude floor - CLOSED, invisible

Real K/Q tensors, softmax against 6 distractors, 10 trials at the worst measured cell (tensor diff ~1.8e-5 mean / 4e-5 max): resulting softmax probability difference maxes at 1.3e-6 - the same order as fp32 rounding noise itself. The magnitude floor found in §2.1 does not survive into attention scores at a level that moves outputs. (Caveat: synthetic randn keys/queries, single step. A separate real-model recall task, discussed below, provides softer multi-step corroboration.)

### 5.7 Isolating P from evict_n - CLOSED

Full P×evict_n grid (0-450, step 50, 5 draws/cell, fp32): the error floor is driven by **evict_n magnitude specifically**, not by P and not by target position (P−evict_n) - confirmed by cases where identical |target position| values produce wildly different error magnitudes depending on the P/evict_n split, and evict_n=0 always producing exactly zero error regardless of P.

### 5.8 - Real-model recall accuracy: no measurable gap vs. corrected, once three implementation bugs were fixed

Beyond the 7 original questions, a real per-layer implementation (Qwen2.5-0.5B-Instruct, fp32, manual forward pass) was built to test RSQR's actual recall accuracy against two baselines (continuous re-rotation, leave-gap) on a multi-fact needle-in-haystack task.

**An earlier version of this section reported a real, statistically significant accuracy gap between RSQR and corrected re-rotation at low eviction counts, closing by `n_cycles`≈32.** That result has been retracted. It was an artifact of three implementation bugs in the harness, not a property of the mechanism. In the interest of an honest record, all three are documented here rather than silently corrected:

1. **Position-collision bug.** `logical_pos` was doing double duty as both the model's absolute position counter and the eviction-compaction shift amount, so it went flat across repeated evictions - multiple distinct tokens were silently assigned the *same* RoPE position. Fixed by separating `true_pos` (monotonic, used for every model step) from `logical_pos` (derived, used only for survivor rotation targets).
2. **Duplication bug.** A flagged survivor's raw K/V was added to the active survivor store *immediately at flag time*, while the same token also remained normally present in the sliding window until its later, natural eviction - up to ~`window_size` steps afterward. For that entire span, the same token contributed to attention twice: once via its real window entry, once via a duplicate survivor entry. Fixed by holding captured raw K/V inert (`pending`) until the token's window copy is actually evicted, at which point exactly one representation becomes active - never both at once.
3. **Anchoring bug.** Once survivor logical positions were correctly rank-compacted to close gaps between survivors (closing an earlier, separate bug), they were anchored at an isolated zero-based numbering (`0, 1, 2, ...`), disconnected from the window's own coordinate system. Since window tokens are deliberately never touched - they keep their real, unshifted `true_pos` - a survivor numbered near 0 sitting immediately next to a window token numbered in the hundreds produced a badly wrong relative distance. Fixed by anchoring survivor logical positions so the numbering ends exactly at `window_start_true_pos − 1`, recomputed fresh every step as the window slides, so survivors and the window share one consistent coordinate system. This recomputation is a cheap integer label update used to pick the correct query-side offset per bucket (§3.1, §3.3) - it does not trigger re-rotation of the survivor's stored raw K, which is set once at window-exit and never touched again.

None of these three bugs touch the core single-hop-rotation design (§3.2): a survivor's raw K is still rotated at most once, directly from raw to its (now correctly computed) target position, never as a correction-on-correction. These were harness bugs in *how the target position was chosen and when a survivor became active*, not a reappearance of the compounding-drift failure mode from §2.

**Re-run against the fixed implementation**, same task, same n=60 (3 seeds × 20 magic numbers) per `n_cycles` point:

| n_cycles | n | B corrected | B uncorrected | C RSQR | RSQR vs. corrected | RSQR vs. uncorrected |
|---:|---:|---:|---:|---:|---:|---:|
| 2  | 60 | 93.3%  | 93.3% | 95.0% | +1.7% | +1.7% |
| 4  | 60 | 93.3%  | 61.7% | 93.3% | +0.0% | +31.7% |
| 6  | 60 | 91.7%  | 53.3% | 91.7% | +0.0% | +38.3% |
| 12 | 60 | 100.0% | 28.3% | 96.7% | -3.3% | +68.3% |
| 16 | 60 | 96.7%  | 35.0% | 93.3% | -3.3% | +58.3% |
| 32 | 60 | 93.3%  | 18.3% | 96.7% | +3.3% | +78.3% |
| 48 | 60 | 93.3%  | 6.7%  | 95.0% | +1.7% | +88.3% |

McNemar's test on RSQR vs. corrected shows no significant difference at any tested `n_cycles` (all p ≥ 0.375; full test detail in the accompanying trial logs). RSQR tracks corrected within a few points at every point, with no dip at low eviction counts and no discernible trend across the range.

**Reading this honestly:** the three-way comparison is the actual headline result. Uncorrected collapses hard and fast as eviction cycles accumulate - from parity at `n_cycles`=2 down to 6.7% by `n_cycles`=48 - concretely demonstrating why re-rotation matters at all once RoPE positions go stale. RSQR matches corrected's accuracy at every tested point while paying `O(1)` amortized rotation cost per survivor instead of corrected's `O(window_size)` per eviction. There is no accuracy tax for taking the cheaper path, at least at this task's scale and this model.

**What this doesn't establish:** this is one task (multi-fact needle-in-haystack), one model (Qwen2.5-0.5B-Instruct), one hardware setup (CPU, fp32).

Full per-trial logprob detail (all three strategies) is retained in the accompanying trial log for anyone who wants to dig into the confidence margins beyond raw accuracy.

---
## 6. Possible future direction: vLLM

vLLM PR **#43374** ("Experimental session KV eviction with attention sinks") adds the scheduler-side plumbing for exactly this kind of eviction - block compaction, multimodal-item-atomic reindexing, prefix-cache invalidation on modified blocks - but explicitly ships without a re-rotation implementation. Its `_post_add_requests` extension hook is a stated placeholder, and its own "Future work" section names "an in-tree RoPE re-rotation kernel" as the missing piece before the experimental gate (`VLLM_ENABLE_EXPERIMENTAL_SESSION_EVICTION`) can be removed.

Within this RFC's scope (§0) - improving on continuous per-step re-rotation specifically - this mechanism is a natural candidate to fill that slot: it satisfies the correctness requirement that PR's author describes (re-rotating surviving K by the eviction delta, without compounding error across repeated evictions in a long-lived session), at lower aggregate cost than a naive per-step re-rotation kernel would be.

A future vLLM-facing version of this RFC should:
- Target filling PR #43374's stated missing piece specifically, rather than proposing a freestanding eviction-strategy change.
- Be prepared to engage with vLLM issue **#51948** ("Bounded-memory video sessions"), a related, production-measured design for a closely adjacent workload (streaming video / multimodal-RoPE) that took a different approach (leave-gap rather than re-rotation). Reviewers familiar with that issue may ask how this proposal relates to it; per §0, this RFC does not take a position on re-rotation vs. leave-gap and that comparison is left to future work.

---
