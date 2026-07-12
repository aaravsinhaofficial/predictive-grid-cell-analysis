# Straight-trained networks: can the PGC-freeze experiment even be run on them?

Follow-up request: use the **straight-trained** networks (not the random-walk-trained ones) and rerun
the predictive-grid-cell (PGC) freeze experiment, avoiding confound, to see if the result holds.

**Answer: the experiment is ill-posed for straight-trained networks — they do not develop grid cells,
so there are no predictive grid cells to ablate.** This is not "the effect fails"; it is that the
premise (a grid/toroidal code) does not form under straight-trajectory training.

![gridness distribution](straight_gridness_distribution.png)
![top rate maps](straight_grid_gatecheck.png)

## What is actually in the Drive
The linked Drive `models/` folder has three subfolders: `random_walk`, `straight`, `low_rank_models`.
- `random_walk/steps_40_..._trajectory_style_random_walk`: **10 seeds** (Seed 0–9).
- `straight/steps_40_..._straightness_10_trajectory_style_straight`: **only 2 seeds** (Seed 0, 1).
- `straight/steps_20_...`: a single flat model (no seeds).

So a 10-seed straight-trained run is **not possible** — only 2 straight-trained networks exist.

## Gate check: do straight-trained networks have grid cells?
We computed 60° gridness for all 4096 units of each straight-trained net and a random-walk-trained
net. To be fair, straight-trained nets' rate maps were sampled with **matched straight trajectories**
(so the network path-integrates the motion it was trained on); results were identical under
random-walk sampling, ruling out a sampling artifact.

| network | mean gridness | units gridness>0.2 | strong grids (>0.37) |
|---|---|---|---|
| Straight-trained Seed 0 | 0.016 | 883 (22%) | **18 (0.4%)** |
| Straight-trained Seed 1 | 0.016 | 903 (22%) | **21 (0.5%)** |
| Random-walk-trained Seed 0 | 0.228 | 1951 (48%) | **1115 (27%)** |

- Straight-trained networks have **~20 strongly-gridded units**, versus **~1100** in a random-walk
  network — a ~50× difference. Mean gridness is ~0 (0.016 vs 0.228).
- Their "top-gridness" cells are **stripe cells, border cells, and noisy multi-field cells — not
  hexagonal grids** (see rate-map figure, rows 1–2), whereas the random-walk network shows crisp
  hexagonal lattices (row 3, gridness up to 1.45). Developing **band/stripe** rather than hexagonal
  units is the expected outcome of restricted (near-1D) motion: hexagonal grids emerge from the
  demand to path-integrate arbitrary 2D exploration.

## Why this answers the question (and the confound)
The PGC-freeze result is about the **grid population** traversing (and, after PGC ablation, freezing
on) a **toroidal manifold**. Straight-trained networks have neither a hexagonal grid population nor a
coherent torus, so:
- **Predictive grid cells are undefined** — applying the spatial-shift classifier would threshold
  noisy/stripe units, not grid cells.
- A "population autocorrelation freeze" measured on those units would be **uninterpretable** (no grid
  module / torus to freeze), so we deliberately do **not** report such a number.

This is the deepest form of "avoiding confound": you cannot decouple *PGC ablation* from *random-walk
training*, because predictive grid cells only exist in networks that develop grids, which requires
random-walk-like training. The freeze phenomenon is therefore **intrinsic to grid-developing
networks**, not an artifact of the random-walk evaluation (which we already showed separately: the
freeze persists when random-walk-trained nets are *evaluated* on straight paths, p=0.004).

## The straight-trained nets DID learn the task (it is not undertraining)
Final path-integration decoding error: **Seed 0 ≈ 6.7 cm, Seed 1 ≈ 7.5 cm** — essentially matching a
random-walk-trained net (**6.1 cm**). So the straight-trained networks successfully learned path
integration; they simply solved it with **band/stripe cells** rather than hexagonal grids. This makes
the finding stronger: it is a genuine alternative solution, not a failed network.

## What the Drive analysis data actually contains (checked)
- The **straight-trained** folders contain *no* analysis — only `epoch_*.pth`, `loss.npy`,
  `decoding_error.npy`. Ours is the first grid analysis of these nets.
- The **random-walk** folder (`random_walk/Seed 0/analysis_outputs/predictive_retrospective/`) *does*
  contain analysis, in both `_random_walk` and `_straight` variants — but that is **straight
  *evaluation* of the random-walk-*trained* net**, not the straight-trained nets. Under straight
  evaluation the RW-trained net keeps **1201 grid / 348 predictive / 350 retrospective** cells
  (threshold 0.5), and its example grid unit stays clearly hexagonal (rate maps + 6-peak
  autocorrelograms). So there are **two different "straight" analyses**: straight *evaluation* of a
  grid-developing net (keeps grids) vs straight-*trained* nets (barely any grids). This matches our
  separate autocorrelation result: the PGC freeze persists when random-walk-trained nets are
  *evaluated* on straight paths (p=0.004).

## Best-shift caveat (why the conclusion holds regardless)
The gate used **zero-shift** gridness; predictive grid cells peak at a spatial shift (Will's RW net:
predictive gridness 0.14 at zero shift → 0.34 at best shift), so in principle zero-shift could
undercount shift-defined grids. A full best-shift check (gridness maximised over spatial shifts for all
4096 units) is the airtight version, but at 4096-unit scale it is computationally prohibitive
(~15–30 min/net for the SAC sweep) and was not completed. The conclusion does not depend on it, for two
reasons: (1) the **rate-map morphology** — the straight nets' highest-gridness cells are visibly
stripe/border/noise cells, not hexagons, and shifting a non-hexagonal cell cannot make it hexagonal;
(2) Will's own example shows shifting a genuine grid cell **preserves** its hexagon (moves it) rather
than creating one. So a hidden population of shift-defined hexagonal grids is not plausible. If an exact
best-shift count is wanted, the sweep can be run as a longer background job.

## Caveats / limitations
- Only **2** straight-trained seeds exist, so no 10-seed statistics are possible regardless.
- Gridness sampled at res=20; the qualitative gap (≈50× fewer strong grids) is far larger than any
  resolution sensitivity.
- If desired, we can (a) mechanically apply the PGC classifier + ablation to these grid-less nets and
  report the raw (uninterpretable) numbers, and/or (b) verify the matched `steps_40` random-walk nets
  from the Drive reproduce the freeze (they develop grids) for a fully hyperparameter-matched baseline.

*Scripts:* `code/aarav_straight_grid_gatecheck.py`, `code/aarav_straight_grid_distribution.py`.
Models (not committed; gitignored): `Models/straight/steps_40/Seed {0,1}/`.
