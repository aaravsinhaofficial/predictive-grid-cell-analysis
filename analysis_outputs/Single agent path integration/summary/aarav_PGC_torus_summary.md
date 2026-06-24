# What are predictive grid cells (PGCs) doing on the torus?

A consolidated, quantified summary of the crossing + ablation experiments (single-agent
path-integrating RNN, all-units / 4096-unit classification, seeds 0–9).

---

## 1. Two different questions about the torus

A trained grid-cell RNN's population activity lives on a **torus** (a donut-shaped manifold in
the 4096-dim activity space). Each instant of activity is one point on the donut, described by
two angles: **θ1** (around the big loop) and **θ2** (around the tube). As the animal moves, the
point should glide smoothly around the donut.

Two *separate* things can break:

| concept | plain English | what it means on the donut | metric (low = healthy) |
|---|---|---|---|
| **Geometry** ("keeping activity ON the torus") | is the donut still there, and does activity stay on its surface? | the donut keeps its shape; states don't fly off into never-visited regions | **ring spread** (radius CV) and **off-manifold distance** (kNN to the cloud of normal states) |
| **Traversal** ("moving activity AROUND the torus") | as you move, does the activity flow smoothly all the way around the ring, or get stuck/bunched at some angles? | θ1 sweeps the full circle vs clumping at a few angles | **θ1 clumping** (resultant length of θ1: 0 = uniform flow, 1 = stuck at one angle) |

Geometry is about the *map existing*; traversal is about *moving along the map*. These are
what "grid cells define the structure / predictive cells move along it" would predict — and we
can test each separately.

---

## 2. Methodology (how each number is produced)

**Get the torus.** Compute each grid cell's spatial rate map; take the FFT of the average map
to recover the grid's two reciprocal-lattice vectors (k1, k2) and each cell's phase. This gives
a fixed linear map: population activity → (θ1, θ2, radius). A coherent single grid **module**
(~389 cells/seed) is detected via UMAP+clustering and the basis is fit on it, so the projection
is a clean donut rather than a multi-module blur.

**Ablate = silence units.** Set a unit's input, recurrent, and output weights to zero (its
activity becomes exactly 0). Every comparison is **count-matched**: the same number of units N
is removed in each condition, so differences reflect *which* cells, not *how many*.

**Metrics.** Run the network on random walks; project activity onto the fixed torus.
- *ring spread* = CV of the major-loop radius (geometry).
- *θ1 clumping* = resultant length of θ1 (traversal).
- *off-manifold* (full population) = mean kNN distance from the ablated endpoint to a cloud of
  normal intact states, in PCA space (geometry; "is this a legal state?").
- *decode error* = error of the network's decoded position (m).
- *effective time* = which timestep of the **intact** path the ablated endpoint best matches
  (1 = advanced correctly; ~0 = state collapsed back toward the start).

**Controls for the predictive/module overlap.** ~123 of the ~389 module cells are predictive,
so "ablate predictive" partly = "ablate module cells." Three nested tests:
1. **Naive**: Predictive(N) vs Random(N).
2. **Off-module**: predictive cells *outside* the module vs random off-module cells (removes any
   module overlap — but also removes the cells most likely to matter).
3. **Within-module (decisive)**: among module cells, Predictive vs Non-predictive, count-matched
   — does predictive coding add anything *beyond* being a torus cell?

---

## 3. Results, quantified (10 seeds, mean ± between-seed std; paired Wilcoxon)

### Geometry — held up by grid cells, NOT predictive cells
| condition (count-matched) | ring spread | off-manifold (×intact) |
|---|---|---|
| Intact | 0.44 ± 0.09 | 1.0 |
| Predictive (N) | 0.61 ± 0.16 | 60 ± 80 |
| Random (N) | 0.70 ± 0.06 | 89 ± 60 |
| Grid (N) | 0.94 ± 0.12 | 193 ± 122 |

- **Grid vs Random**: ring spread Δ +0.24, **10/10, p = 0.002**; off-manifold Δ +104×, **10/10, p = 0.002**. Grid cells robustly hold the geometry.
- **Predictive vs Random**: predictive *preserves* shape (ring spread Δ −0.09, 3/10) and is not significantly more on-manifold (off-manifold p = 0.23, and that metric swings 100× across seeds — unreliable). **Predictive cells are not structural.**

### Traversal — predictive cells contribute, but weakly
| condition | θ1 clumping (low = flows) |
|---|---|
| Intact | 0.22 ± 0.14 |
| Predictive (N) | 0.49 ± 0.24 |
| Random (N) | 0.30 ± 0.17 |
| Predictive **in-module** (N″) | 0.41 ± 0.20 |
| Non-predictive **in-module** (N″) | 0.32 ± 0.10 |

- **Naive** Pred(N) vs Rand(N): Δ +0.20, **8/10, p = 0.010**. Suggestive, but confounded by module overlap.
- **Off-module**: Δ +0.04, 5/10, **p = 0.49 (n.s.)** — predictive cells outside the module don't matter.
- **Within-module (decisive)**: Pred-in vs NonPred-in Δ +0.09, **7/10, p = 0.049** — predictive coding adds a **weak but real** traversal effect *beyond* module membership.

### Full-population dynamics
| | Predictive(N) | Random(N) | Structural-grid(N) |
|---|---|---|---|
| decode error (m) | **1.11** | 0.96 | 1.00 |
| effective time | 0.07 | 0.07 | 0.08 |

- **Decode error**: Predictive highest, Δ +0.15 m vs Random, **10/10, p = 0.002**. A *frozen-decoder control* (decode the ablated activity with the original intact decoder) gave **identical** errors (max diff 0.0 m) → this is a genuine loss of position information in the state, **not** a readout artifact. Predictive cells carry the most position information.
- **Effective time**: *every* ablation collapses the state back toward the start (~0.07 vs intact 1.0), predictive ≈ random (**p = 0.70**). So "predictive specifically advances the state" is **not** supported — all ablations stall integration equally.

### Crossing (Exp 1, path-dependence; seeds 0–1)
Designed X-crossings and William's same-bin both show predictive units **decorrelate the most**
when position is fixed but heading differs (X-crossing correlation 1.00 → 0.89 over 10°→170°,
vs standard grid 1.00 → 0.95). Predictive units are robustly **path/heading-dependent**.

---

## 4. So what role are PGCs playing on the torus?

- **They do NOT build or hold the torus (geometry).** That is the broad grid-cell population —
  silencing grid cells wrecks the donut and throws activity off-manifold (robust, 10/10).
  Silencing predictive cells leaves the shape intact.
- **They contribute weakly to traversal (flow around the torus).** The decisive within-module
  control keeps a marginal effect (p ≈ 0.05): predictive grid cells help move activity around
  the ring slightly more than non-predictive grid cells do. Borderline, not strong.
- **They carry position information.** Removing them degrades the network's decoded position more
  than removing random cells (robust, 10/10) — consistent with their path/heading-dependence.
- **They are NOT uniquely responsible for "advancing the state."** Under any ablation the state
  fails to advance equally; predictive cells aren't special on that axis.

**One-line role:** predictive grid cells are a path/heading-dependent, position-informative
subset that makes a *weak, borderline-significant* contribution to moving activity around the
torus — they are **not** what maintains the torus, and the strong "they drive the dynamics"
story does not survive rigorous count-matched testing.

## 5. Caveats
- The within-module traversal effect is right at p = 0.05 (7/10 seeds; 3 go the other way; effect
  ≈ its own variance) and we ran many tests this session — treat as *suggestive*, not established.
- The off-manifold metric is wildly seed-variable (100× swing); geometry conclusions lean on
  *ring spread* (tight) and the structural-vs-random off-manifold contrast (robust).
- Ablation is a blunt tool; a positive **intervention** (drive/freeze the predictive subspace and
  test whether decoded position advances) would convert these null/weak ablation results into a
  direct causal test.

Figures: `torus_ablation_metrics_across_seeds.png` (geometry + traversal + the 3 controls),
`predictive_dynamics_across_seeds.png` (off-manifold / decode error / effective time),
`torus_ablation_visual.png` (the tori), `../aarav_crossing_population_corr/…` (path-dependence).
