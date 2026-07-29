# Sparse Separable Memory for Sequence Models

Professor-facing presentation outline
Kernel-Adaptive Memory (KAM), Phase VI
Prepared July 28, 2026

## Design and speaking intent

- Length: 9 slides, approximately 10–12 minutes plus discussion.
- Audience: technically literate professor who does not need implementation detail on every slide.
- Narrative: motivation → intuition → one architecture slide → research variants → evidence standard → preliminary results → decision paths.
- Visual language: off-white background, deep navy text, restrained blue and amber accents, diagrams and direct labels instead of decorative imagery.
- Evidence boundary: the three-seed pilot is directional only. The fixed-sample confirmation campaign must finish before its scientific effects are inspected.

---

## Slide 1 — Sparse Separable Memory for Sequence Models

Subtitle: Can a sequence model gain large, adaptable memory without activating all of it for every token?

Footer: Kernel-Adaptive Memory (KAM) · Phase VI

Speaker cue: The question is not whether memory replaces attention. It is whether a sparse memory can complement a strong sequence model under fair compute and parameter comparisons.

---

## Slide 2 — The capacity–computation tension

Headline: More stored knowledge usually means more work at inference.

Visual:

1. Dense model: every token passes through the same large block.
2. Sparse memory: every token consults only a few relevant entries.

Key idea: Decouple how much the model can store from how much it must activate.

Speaker cue: Attention is good at relating the current context. The proposed memory is a different mechanism: a selectively accessed bank of reusable local transformations.

---

## Slide 3 — The core intuition: route, retrieve, blend

Visual flow:

`current representation → router → top-K supports → local experts → gated residual`

Three labels:

- Route: identify the small neighborhood relevant to the current state.
- Retrieve: combine a few local values or local affine experts.
- Blend: add the memory contribution conservatively to the Transformer block.

Bottom line: Large potential memory; small active path.

Speaker cue: The zero-initialized gate lets the model begin as its baseline and learn to use memory only when useful.

---

## Slide 4 — One technical slide: sparse separable KAM

Minimal notation:

`I(z) = TopK(similarity(z, keys))`

`memory(u) = Σ over i in I(z) of routing_weight_i × local_expert_i(u)`

`output = Transformer_path + small_FFN + gate × memory`

Fairness constraints:

- Same corpus, tokenizer, optimizer, and token budget.
- Match total parameters and report active parameters/FLOPs.
- Compare against dense, widened, and conditional-memory controls.

Speaker cue: “Separable” means the routing geometry and the returned algebra can be treated as different objects, rather than forcing one optimizer to move everything at the same rate.

---

## Slide 5 — Two timescales: learn the map, then stabilize it

Headline: Where to look should often change more slowly than what to return.

Timeline:

1. Adapt: keys and experts learn useful neighborhoods.
2. Stabilize: geometry updates become small.
3. Freeze: keys stop moving near 80% of training.
4. Final tune: values, experts, gate, and backbone refine against stable routing.
5. Evaluate: validation during training; held-out test only at the locked endpoint.

Analogy:

- Geometry = the library’s shelving system.
- Algebra = the content available at each shelf.

Speaker cue: The confirmation campaign explicitly checks nonzero pre-freeze key gradients and essentially zero post-freeze drift. Fixed-key success does not prove learned geometry helps.

---

## Slide 6 — Current method and the research branches

Current strongest candidate:

- T-KAM-F: fixed or data-centered keys, sparse top-K routing, local experts, gated residual.

Research branches:

- Learned geometry (T-KAM-L): can the support map improve without becoming unstable?
- Alternating optimization (T-KAM-ALT): update algebra quickly and geometry conservatively.
- Online adaptation: update values or experts after a regime change.
- Dual memory: combine slow persistent memory with a small recent episodic bank.
- Efficient routing: exact, chunked, and product-key lookup at larger bank sizes.

Speaker cue: These branches test distinct hypotheses. A win for fixed keys cannot be used as evidence for learned keys, online adaptation, or a particular router.

---

## Slide 7 — What would count as convincing evidence?

Headline: Fair comparisons, held-out data, paired seeds, and a prespecified decision.

Confirmation design:

- 156 fixed rows on four L4 GPUs.
- Primary: 30 paired TinyStories seeds, T-KAM-F versus T-WIDE.
- Replication: 24 paired Tiny Shakespeare seeds.
- Controls: T0 and product-key memory.
- Mechanism: fixed, learned, and alternating KAM variants.
- Primary endpoint: held-out test loss after 50M training tokens.
- Minimum relevant improvement: 2% lower loss.

Pass rule:

Primary and replication must both pass; no optional stopping or seed replacement.

Speaker cue: This is intentionally harder than showing a favorable average. The confidence interval must clear the scientific margin, not merely touch zero.

---

## Slide 8 — Intermediate evidence: promising direction, not a conclusion

Left visual: pilot validation loss at the exact registered checkpoint.

- T-KAM-F: 2.0081
- T-WIDE: 2.8833
- Directional difference: 30.4% lower for T-KAM-F

Under-chart caveat:

- Only 3 paired seeds.
- Exact paired permutation p = 0.25.
- Holm-adjusted p = 0.75.
- Interpretation: effect magnitude is interesting; statistical evidence is insufficient.

Right visual: confirmation campaign status as of July 28, 2026 at 6:40 PM EDT.

- 43 of 156 complete.
- 4 running.
- 109 pending.
- 0 failures.
- Scientific effects remain blinded until the fixed sample is complete.

Speaker cue: The pilot justifies confirmation, not acceleration. The current run was designed to test whether the effect survives cleaner data order, larger corpora, held-out testing, and many more paired seeds.

---

## Slide 9 — The decision after confirmation

Three paths:

1. Robust fixed-key advantage
   Accelerate T-KAM-F; scale memory banks and test larger language settings.

2. Fixed keys help, learned lifecycle fails
   Keep stable/data-centered geometry; stop claiming learned-memory benefits.

3. No replicated quality edge
   Do not position KAM as a general Transformer replacement; retain only supported niches such as adaptation or diagnostics.

Discussion prompts:

- Is a 2% loss improvement the right scientific threshold?
- Should the next scale-up optimize quality, active compute, or adaptation speed?
- Which failure mode would be most informative for the broader research direction?

Closing line: The goal is a narrow, testable claim—not a universal memory architecture claim.

---

## Evidence provenance

- `docs/codex/KAM_PHASE6_SPARSE_SEPARABLE_MEMORY_BRIEF.md`
- `docs/codex/KAM_PHASE6_ANALYSIS_V2_CORRECTION.md`
- `docs/codex/KAM_PHASE6_CONFIRMATION_V2_PREREGISTRATION.md`
- `reports/phase6/PHASE6_ARCHITECTURE_REPORT.md`
- `reports/phase6/PHASE6_OPTIMIZATION_REPORT.md`
- `reports/phase6/PHASE6_ADAPTATION_REPORT.md`
