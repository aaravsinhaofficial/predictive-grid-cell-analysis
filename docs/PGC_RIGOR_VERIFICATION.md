# Verification & adversarial audit of the PGC rigor results

Every headline result was re-derived independently and stress-tested with the
control that would expose it as an artifact. Verdicts and honest caveats below.

## Integrity
- Headline numbers re-derived directly from the raw per-seed files (bypassing the
  aggregators) match every reported figure exactly.
- Git tree clean; all commits on `origin/pgc-rigor-upgrade`.

## 1. Classifier — HOLDS (one consistency bug found and fixed)
- The shuffle null **discriminates**: grid-class units p ≤ 0.039 vs non-grid
  median p = 0.43 (85.8% of null-tested non-grid units p ≥ α).
- Edge exclusion is load-bearing: 28 edge-flagged units, **0** in any grid class.
- Held-out confirmation active: 106/256 candidates fail it; **0** predictive/
  retrospective labels exist without it; 34 would-be directional units demoted.
- **Bug (fixed):** the relaunched (faster) classify sweep *skipped* seed_0, so
  seed_0 was classified at `n_shuffles=50, gridness_floor=0.10` while seeds 1–29
  used `40 / 0.15`. The cross-seed aggregate mixed params. **Fixed** by re-running
  seed_0's full pipeline at the uniform `40 / 0.15` and re-aggregating; the freeze
  now records one uniform config for all 30 seeds.

## 2. Matched ablation — HOLDS WITH CAVEAT
- Effect reproduces (predictive target 0.139 vs matched 0.045, Wilcoxon p=2.3×10⁻⁶),
  deterministic on re-run, and the pool is adequate (median 2.3× target).
- Matching genuinely works: mean |SMD| across all covariates/seeds = **0.144
  (matched) vs 0.259 (random)**; every covariate's *mean* |SMD| < 0.25.
- **Caveat:** two covariates stay imbalanced — **firing-rate variance** and
  **head-direction tuning** — because predictive cells sit in distribution tails
  the retro∪normal pool underpopulates (signed residuals rate_var +0.14, hd_r
  +0.09, predictive-higher; 19/30 seeds have ≥1 covariate |SMD| > 0.25 at the full
  dose). So a fraction of the predictive-vs-matched gap could be attributable to
  those two covariates rather than the predictive shift itself. Fix path: widen the
  pool (match against all units, not just other grid cells) or exact-stratify on
  rate_var/hd_r.

## 3. Path-dependence — HOLDS WITH CAVEAT
- Label-shuffle permutation null (500× per seed, class sizes preserved): true
  cohort gap **+0.00993** vs null mean +0.00135 (97.5th pct +0.0080),
  **permutation p = 0.004**. Within every seed the shuffled predictive and control
  slopes are equal, so class size does not create the gap.
- **Caveat:** the null mean is not zero (+0.00135, ~14% of the observed gap) — a
  small size effect (predictive is a slightly smaller set). ~86% of the gap is
  genuine unit-identity signal. Effect is modest and seed-heterogeneous (23/30
  seeds positive).

## 4. Intervention (flagship) — HOLDS (now with a specificity control)
The rescue injects the predictive ensemble's own downstream recurrent current
(`g_intact[pred] @ W_hh[:,pred]ᵀ`). The missing control was **signal specificity**.
`code/pgc_intervention_specificity.py` runs it on real-gap seeds (0, 2, 20):

| condition | seed 0 | seed 2 | seed 20 |
|---|---|---|---|
| **TRUE PGC replay** | **0.93** | **0.96** | **0.98** |
| SCRAMBLED (same magnitude, shuffled timing) | −0.22 | −0.12 | −0.11 |
| RANDOM subspace (RMS-matched) | 0.25 | 0.26 | 0.09 |

(values = fraction of the ablation decode-error gap recovered.) The scrambled
control is **exactly magnitude-matched** to the true replay yet makes decoding
*worse*; a random-subspace drive of the same energy recovers ≤ 0.26. So the rescue
is **PGC-specific**, not nonspecific energy injection.

**Caveats (about effect size, not specificity):** (a) the "gap" is only
substantial on ~25/30 seeds — on seeds where ablating predictive units barely
damages decoding there is little to rescue, so per-seed recovered-fraction is noisy
there; the cross-seed 86% is meaningful where ablation actually breaks decoding.
(b) The replay pipes the intact network's own current in (a mild teacher-forcing
character), so near-total recovery on real-gap seeds is unsurprising — the *novel*
claim the controls establish is specificity, not the magnitude.

## Not run cohort-wide
Data-driven torus topology (`pgc_torus_topology.py`) was validated on Seed 0 only;
its single-seed Betti readout was noisy (small point cloud) and it has **not** been
run across the cohort — treat it as a framework, not a finished figure.
