# Predictive grid cells — reviewer-grade rigor upgrade

This document tracks the analysis-methodology upgrades built to move the
predictive-grid-cell (PGC) study toward a *Science Advances*-level standard. It
maps each requested item to a concrete implementation in `code/pgc_*.py` and
states its honest status (implemented / partial / pending). Numbers quoted are
preliminary single-seed values used to validate the code; the cross-seed
statistics come from running the pipeline over the full cohort (see
*Reproducibility & cohort*).

All new modules share one foundation (`code/pgc_common.py`) so every per-unit
quantity is derived from **one seeded activation collection** with **one unit
index** (`0..Ng_use-1`), and the gridness definition is the *unchanged*
`GridScorer` (parallelised across cores in `code/pgc_fastscore.py`, not
re-derived). Run the model forward on GPU (`--device cuda`) and set
`OMP_NUM_THREADS=1` so the CPU scoring workers don't oversubscribe.

---

## 1. A defensible PGC classifier — `code/pgc_classifier.py`

The previous classifier existed as two divergent copies
(`classify_from_scores`, `classify_units`) that took a per-unit **argmax of
gridness over a ±20 cm window with no significance test**, so (a) preferred
shifts piled up at the sweep edge and (b) any unit whose gridness merely cleared
0.2 was accepted. The replacement is a single rule with four controls:

- **Wider shift window (stops edge pile-up).** The sweep reaches `±max_lag`
  steps ≈ **±50 cm** at the ~2 cm/step of these trajectories. Units whose optimum
  still sits at the terminal shift are flagged (`edge_flag`) and **excluded**, so
  an unresolved peak can never be reported as a preferred shift.
- **Block / circular shuffle significance.** For each candidate unit a null
  distribution of the best-shift gridness is built by circularly rolling its
  activity in time *per trajectory* (preserving the unit's own temporal
  autocorrelation, destroying the activity↔position pairing) and recomputing the
  max-over-lags gridness. A unit is called a grid cell only if its observed
  best-shift gridness beats the null at `alpha=0.05` (empirical p-value with the
  same max-over-lags selection applied to the null). A block-permutation variant
  is available (`--null_block`).
- **Classify on one trajectory set, confirm on held-out.** Labels and preferred-
  shift *sign* are computed on trajectory set A; the sign must reproduce on a
  **disjoint held-out set B**, with the directional gridness on B also clearing
  the floor. Predictive/retrospective calls that fail this are demoted to
  zero-lag.
- **One classifier** feeds everything downstream (`pgc_classification.npz`).

*Status: implemented and validated, run across the full 30-seed cohort.*

**Cross-seed result (canonical 30-seed cohort, mean ± SEM over seeds, 256 units,
40 shuffles, ±50 cm window):**

| class | % of all units | preferred shift |
|---|---|---|
| Predictive | **9.62% ± 0.38%** (30.7% ± 1.1% of grid cells) | **+21.2 ± 1.2 cm** |
| Retrospective | 5.52% ± 0.31% | −23.9 ± 1.3 cm |
| Zero-lag grid | 16.15% ± 0.48% | ~0 |
| Grid total | 31.29% ± 0.53% | — |

Predictive grid cells emerge in **every** seed (per-seed range 6.2–14.5%), which
is the "prevalence across networks" result. Aggregation + figure:
`analysis_outputs/canonical_cohort_v1_pgc_rigor_summary/`
(`pgc_prevalence_across_seeds.png`, `pgc_classification_across_seeds.json`).

---

## 2. Properly matched causal controls — `code/pgc_matched_ablation.py`

The only prior control was a count-matched *random* draw. The upgrade matches
each ablated PGC to control units on the covariates a reviewer would demand,
assembled per unit in `code/pgc_covariates.py` from one collection:

> module (grid-spacing cluster), gridness, bandness, firing-rate variance,
> head-direction (movement-direction) tuning, decoder-weight magnitude, incoming
> and outgoing recurrent strength — plus exact **unit-count** matching.

- **Matched sampler**: standardise covariates over the eligible pool, greedily
  pick each target unit's nearest unused pool unit in covariate space (optionally
  exact-matching module label); count matches the ablated set exactly.
- **Retrospective and zero-lag grid cells** are first-class ablation groups
  alongside predictive, each with matched + random controls.
- **Complete dose–response + pre-ceiling error.** Error is reported vs *actual
  units removed* (k = 1, 2, 4, …, up to the class size), not a class-size-
  dependent percentile. A **chance ceiling** is estimated by permuting decoded vs
  true positions, and a **pre-ceiling slope** (cm per unit removed) is fit over
  the doses below 0.9×ceiling — so the low-dose regime, where PGC ablation is
  *not yet* uniquely damaging, is reported honestly rather than hidden by
  saturation.

*Status: implemented and run across the full 30-seed cohort.*

**Cross-seed result (30 seeds, pre-ceiling slope = cm error per unit removed,
mean over seeds; paired target-vs-matched test):**

| class ablated | target slope | property-matched | random | target vs matched |
|---|---|---|---|---|
| **Predictive** | **0.139** | 0.045 | 0.097 | paired-t p=0.005, **Wilcoxon p=2×10⁻⁶** |
| **Retrospective** | 0.103 | 0.040 | 0.096 | paired-t p=0.003, Wilcoxon p=5×10⁻⁴ |
| Zero-lag grid | 0.107 | 0.109 | 0.140 | p=0.92 (**not** selective) |

Predictive *and* retrospective ablation is ~3× more damaging than a control
matched on all 7 covariates, whereas zero-lag grid cells are indistinguishable
from their matched controls — a clean selective-importance dissociation. Figure +
stats: `analysis_outputs/canonical_cohort_v1_pgc_rigor_summary/pgc_matched_ablation_across_seeds.{png,json}`.
Chance ceiling ~118 cm; baseline ~4.8 cm.

---

## 3. Direct evidence that PGCs control toroidal flow

### 3a. Data-driven topology — `code/pgc_torus_topology.py`
The prior torus was **assumed**: `estimate_lattice_vectors` took one FFT peak and
*forced* the second lattice vector to be a +60° rotation (hexagonal), and phases
came from that assumed basis. The upgrade tests the topology from the data:

- **Betti verification** via `ripser(maxdim=2)` on a subsampled population point
  cloud, counting persistent H1 bars (a torus expects **2**) and H2 (**1**)
  against a **shuffled null** that sets the significance threshold.
- **Data-driven circular coordinates** lifted from the two longest-lived H1
  cocycles (`do_cocycles=True` + harmonic least-squares smoothing), giving two
  phase coordinates that do **not** assume hexagonal geometry; validated by
  circular independence and by how well they predict physical position.

### 3b. Phase velocity/displacement after intervention & 3c. rescue — `code/pgc_intervention.py`
A manual, per-timestep RNN rollout (validated to match `model.g` to ~1e-4 with
zero drive) lets activity be injected along the **PGC subspace**:

- **Stimulate / replay**: drive the intact network along the PGC subspace and
  measure induced **toroidal phase velocity** and net **phase displacement**;
  drive magnitude → monotonic phase displacement (dose curve).
- **Rescue**: ablate predictive units → the torus phase freezes and decoding
  error rises; then replay the intact network's predictive-unit drive into the
  ablated rollout and show phase velocity / decoding error partially **recover**.

*Status: implemented (agent-built, verified). This is the flagship, potentially
Science-Advances-making experiment; effect sizes are reported as measured,
including partial rescue.*

---

## 4. Replication / statistics package

- **Path-dependence across all seeds** — `code/pgc_pathdep_allseeds.py`: the
  "predictive units encode future trajectory, not just position" test (activity
  divergence vs heading separation at matched locations, predictive vs zero-lag
  control) run per seed and aggregated across the whole cohort with a paired test
  — not the previous 2-seed version.
- **Analyse across seeds/modules, not pooled cells**: the classifier and
  covariates carry per-unit module labels; cross-seed aggregation keys on the
  seed path token and reports mean ± SEM with per-seed points.
- **Reproducibility freeze** — `code/pgc_freeze.py`: records git commit + dirty
  state, full `pip freeze`, per-checkpoint SHA-256 + inferred training config, and
  the exact analysis config (`ClassifierConfig`, covariate defaults) into
  `reproducibility/manifest.json` + `FREEZE.md`, with a caller-supplied
  deterministic timestamp.
- **30 seeds** — a fresh, fully **seed-controlled** cohort
  (`Models/canonical_cohort_v1/`, `run_train_cohort.sh`) trained under one frozen
  recipe (see `canonical-cohort-v1`), because the legacy 10 seeds have no recorded
  recipe/seed and mixing recipes would be a reviewer red flag. `main.py` gained a
  `--seed` flag (training was previously nondeterministic).

*Neural-data items (cross-animal biology; PGC predicting next toroidal phase in
recordings) are intentionally out of scope here — no MEC recordings are present
in this repo. The harness is model-side; drop in a dataset to extend it.*

---

## Reproducibility & cohort

- Frozen recipe & cohort: `run_train_cohort.sh` → `code/train_cohort_worker.sh`;
  30 networks, seeds 0–29, 40k steps each, ~4.3–4.6 cm decoding error.
- Full rigor pipeline over a checkpoint list: `run_pgc_rigor_pipeline.sh`
  (classifier → covariates → matched ablation → torus topology → intervention),
  then `code/pgc_aggregate_seeds.py` for cross-seed statistics.
- Run forward on GPU, scoring on CPU (`OMP_NUM_THREADS=1`).
