# Codex Log

This file is a compact research diary for the optimizer experiments in this repo: what was implemented, why it was added, what we learned, and what the current codebase contains.

## Current Optimizer Roster

- `BASE-SGD`
  Plain SGD baseline.
- `BASE-ADAM`
  Plain Adam baseline.
- `BASE-MUON`
  Muon for matrix-like parameters, Adam fallback for embedding-like and scalar parameters.
- `AVG-8`
  Weighted history baseline:
  `m_t = (1 - beta_mix) g_t + beta_mix * mean(g_{t-1}, ..., g_{t-8})`
- `AVG-8R`
  Same as `AVG-8`, but second moment uses raw `g_t^2` instead of mixed-history `m_t^2`.
- `ATTNRAW-8`
  Parameter-free Type 1 attention over the last 8 past gradients:
  `m_t = (1 - beta_mix) g_t + beta_mix * attn_raw(past)`
- `ATTNRAW-8R`
  Same as `ATTNRAW-8`, but second moment uses raw `g_t^2`.
- `ATTNRAW-V2-8`
  Recursive attention variant:
  `m_tilde = attn([g_t, g_{t-1}, ...])`, then Adam-style EMA on top.
- `ATTNRAW-V3-8`
  Past-only attention with current-gradient residual, then EMA on top:
  `u_t = (1 - lambda) g_t + lambda * attn(past)`
- `ATTNOPT-B-8`
  Learned Type 2 variant trained by next-gradient prediction.
- `ATTNOPT-A-8`
  Learned Type 2 variant trained by differentiating a virtual step against a held-out batch.

## Naming History

- Early drafts used names like `AttnOpt`, `SlidingAvg`, and `Type 1`.
- The current repo split things more explicitly:
  - `AttnRaw` = parameter-free tensorwise attention
  - `Avg` = dumb history baseline
  - `AttnOptA/B` = learned projection variants

## Mini Blog

### 1. Start With Ordinary Baselines

The project started with the obvious controls: `SGD`, `Adam`, and `Muon`.

The point was simple: if a history-based optimizer could not at least compete with normal adaptive baselines, the idea was not worth much. `Muon` stayed in the matrix because it is an interesting geometry-aware baseline for transformer-style weight matrices.

### 2. The First Honest Objection: Why Not Just Average?

The first real criticism was the right one: if past gradients are all roughly useful, why not just average them? That led to the uniform-history baseline.

This became `Avg`, which is not learned and not selective. It asks a blunt question:

If attention is doing anything real, can it beat a cheap sliding-window memory?

That was an important framing change. The comparison stopped being only against Adam and started being against the stronger conceptual baseline: uniform retrieval from recent history.

### 3. Type 1: Raw Tensorwise Attention

The first nontrivial optimizer was `AttnRaw`.

Implementation:
- One history buffer per parameter tensor.
- Flatten the current gradient and each stored past gradient.
- Score past entries by cosine similarity with the current gradient.
- Softmax the scores.
- Retrieve a weighted sum of past gradients.
- Mix that retrieved past with the current gradient.
- Feed the result into Adam-style normalization.

This was the first version that answered the “why not average?” criticism with actual code instead of rhetoric.

### 4. Second-Moment Ablations

Once `Avg` and `AttnRaw` existed, the next question was whether the second moment should use:

- raw `g_t`, or
- the final mixed update direction

That created the `R` variants:
- `AVG-8R`
- `ATTNRAW-8R`

This mattered because if the numerator is already smoothed by history, building the denominator from that same smoothed object may be double-smoothing the optimizer.

### 5. First Empirical Signal

The first important result was not “attention beats Adam.”

It was:
- `ATTNRAW-8` clearly beat `AVG-8`

That mattered a lot. It meant the recent gradient history was not behaving like a homogeneous bag of equally useful memories. Selective retrieval had signal.

From the current `logs/` directory, the clearest completed early-run comparison is:

- `AVG-8`: tail-20 avg loss `4.2057`
- `AVG-8R`: tail-20 avg loss `3.9738`
- `ATTNRAW-8`: tail-20 avg loss `3.8057`
- `ATTNRAW-8R`: tail-20 avg loss `3.8695`

So raw attention beat averaging, and the raw-second-moment ablation helped the averaging baseline a lot.

### 6. Recursive Variant V2

Then came the idea: maybe the real weakness of `AttnRaw` is that it only has a fixed local window, while Adam has recursive memory. So `AttnRawV2` was created.

The idea was:
- include `g_t` directly in the attention window
- let attention produce an “instantaneous signal”
- run EMA on top of that signal, Adam-style

Conceptually, this was supposed to combine:
- attention for local retrieval
- EMA for long memory

In practice, this ran into the obvious problem: if `g_t` is in the attention pool, it can dominate by attending to itself. Empirically, this variant performed badly.

Current on-disk run:
- `ATTNRAW-V2-8`: tail-20 avg loss `4.6981`

That is a useful negative result. “Just include the current gradient inside raw attention and recurse” does not seem to be the right design.

### 7. Recursive Variant V3

To fix the self-dominance problem, `AttnRawV3` was added.

The new structure was:
- `g_t` is query only
- attention retrieves from past gradients only
- current gradient is added back through an explicit residual path
- EMA then runs on top

This is cleaner than V2, because attention no longer has an easy collapse-to-self route.

The current config uses `mix_beta = 0.1` for `ATTNRAW-V3-8`, which means the current gradient dominates and the attended past is only a correction term.

Current on-disk run:
- `ATTNRAW-V3-8`: tail-20 avg loss `5.1198`

So the current V3 setup is not yet good. That does not kill the idea, but it means this particular recursive formulation is not there yet.

### 8. Type 2: Learn the Similarity Space

Once `AttnRaw` showed that selective retrieval beat averaging, the next natural step was to stop using raw cosine and instead learn the query/key projections.

That produced two learned variants:

- `AttnOptB`
  Learn `W_Q` and `W_K` by asking whether the previous step’s `m_tilde` predicts the next observed gradient.

- `AttnOptA`
  Learn `W_Q` and `W_K` by asking whether the current train-step update would have produced a better held-out loss after a differentiable virtual step.

These are heavier and more meta-learning flavored than the raw variants.

### 9. Repairing the Learned Variants

The first pass of `AttnOptA/B` had multiple implementation problems:

- `d_attn` was not actually the projection width
- attention included the current gradient inside history, contrary to the intended math
- `AttnOptA` used `.data` swapping, which broke the differentiable meta-step

Those were fixed.

Current design:
- `AttnOptB`
  - uses real `W_Q, W_K in R^{d x d_attn}`
  - attends over past history only
  - forms `m_tilde = (1 - mix_beta) g_t + mix_beta * attn(past)`
  - updates `W_Q/W_K` by next-gradient alignment

- `AttnOptA`
  - uses the same `m_tilde` construction
  - stores the pre-step parameter/state snapshot
  - rebuilds the exact current step differentiably
  - evaluates a held-out batch through `torch.func.functional_call`
  - backprops held-out loss into `W_Q/W_K`

This makes `AttnOptA` an actual bilevel-style optimizer, at least mechanically.

## Current Read On The Project

The strongest empirical claim so far is not that learned attention wins.

It is this:

Parameter-free selective retrieval over recent gradients beat uniform averaging over the same history window.

That is already enough to justify continuing the project. It means there is real structure in gradient history beyond “just average the last few steps.”

The unresolved question is what the best next layer of complexity should be:

- better recursive variants
- better second-moment choices
- or learned Type 2 attention

## Notes On Logs

- `logs/BASE-ADAMW/metrics.jsonl` is a historical leftover from an earlier phase. The repo no longer uses AdamW in the current training path.
- `logs/BASE-ADAM/metrics.jsonl` is currently shorter than the other runs, so its on-disk comparison is incomplete.
- The most complete current comparison from logs is therefore between `AVG-*`, `ATTNRAW-*`, `MUON`, and `SGD`.
