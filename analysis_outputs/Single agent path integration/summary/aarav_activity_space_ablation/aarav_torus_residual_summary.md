# Where does the predictive-ablated "wandering" come from? Torus vs residual, and grid aliasing

Follow-up to the output-space result (PGC ablation makes the decoded position *wander*, not clump).
Two questions, 10 seeds, all 4096 units, seq-len 40:
1. **Is the wandering ON the torus or OFF it?** Decompose the decoded output into a torus part and
   an off-manifold residual; retrain a decoder on ablated activity; measure traversal, weights,
   drift vs diffusion.
2. **Are the "different" wander locations actually the same place modulo the grid lattice?**
   (Rob's hypothesis: one torus phase = many physical locations = a grid cell's fields.)

![decomposition](torus_residual_decomposition.png)
![aliasing](torus_aliasing.png)

## 1. The wandering is OFF the torus; the position code collapses

- **Output wanders, torus is frozen.** The decoded-output spread (gyration) is large for predictive
  ablation (~0.7 m) but the *torus-phase path* (position integrated from the module's θ1,θ2 through
  the lattice — periodicity-free) is concentrated (~0.1–0.3 m, *below* intact). So the wandering is
  **not** in the torus angles — it lives in the off-torus residual. This reconciles the apparent
  contradiction with the latent θ1 "clumping": on the torus the activity *does* clump/stall; in the
  output it wanders, because the wandering is the off-manifold part the decoder reads.
  *(Note: the literal `D(m(θ))` reconstruction is not used for this — a single torus module is
  periodic and cannot specify absolute position, so `D(m(θ))` is dispersed even for intact. The
  phase-integrated path is the correct periodicity-free torus position.)*

- **The position code genuinely collapses — it is NOT decoder distribution shift.** A *freshly
  trained, cross-validated* decoder recovers position from **intact** activity at ~10 cm (R²≈0.73)
  but from **predictive-ablated** activity only at ~85–90 cm (R²≈0.03) — essentially chance. So the
  original decoder isn't just "looking in the wrong place"; there is no recoverable position signal
  left. **This collapse is generic to ablation** (grid, random-N, structural-N all collapse to
  ~90 cm / R²≈0) — removing ~600 units and integrating through the broken recurrent loop for 40
  steps decorrelates activity from current position history-dependently.

- **Predictive-specific effect: stalled traversal.** Among equal-sized lesions, removing predictive
  cells specifically *reduces* torus traversal (winding number) relative to random-N (p=0.049) — the
  genuine PGC traversal role, consistent with the θ1-clumping result.

- **PGCs are not specially weighted.** Decoder-column norm and outgoing-recurrent-norm of predictive
  cells ≈ structural grid cells (4.51 vs 4.60; 0.179 vs 0.182). So the predictive-specific traversal
  effect is **dynamical / connectivity-pattern**, not "PGCs are high-leverage units."

- **Drift + diffusion.** Predictive ablation shows both a systematic decoded-velocity bias
  (~1–2 cm/step) and growing cross-trial variance (diffusion).

## 2. ~Half of the wandering IS grid-lattice aliasing (Rob's hypothesis, confirmed in part)

Fold each decoded position into one grid-lattice unit cell:
`u = K·p` (lattice coords), `frac = u − round(u)` (within-cell phase), `p_within = A·frac`.
If the wander is jumps between a single phase's grid-field replicas, folding collapses it.

| condition | physical wander (within a trajectory) | folded into one cell | **aliasing index** |
|---|---|---|---|
| Intact | 0.23 m (<⅓ grid period) | 0.22 m | **0.02** (none) |
| **Predictive-ablated** | **0.69 m (~1 grid period)** | **0.31 m** | **0.52** |
| Grid-ablated | 0.39 m | 0.28 m | 0.24 |
| Random-ablated (N) | 0.64 m | 0.33 m | 0.46 |

- **~52% of the predictive-ablated wander disappears when folded into one lattice cell** — those
  "different" locations really are **grid-field replicas of related phases** (the same place mod the
  grid lattice). Predictive > intact aliasing in **10/10 seeds, p=0.002**; predictive > grid 10/10,
  p=0.002. Intact shows ~0 aliasing (it never leaves one cell, so it can't alias).
- **~60% of the large decoded jumps are lattice translations** (fold to ≈0).
- It is **not unique to predictive**: random-N ablation aliases almost as much (0.46) — aliasing is a
  property of "the decoded estimate wandering across ≳1 grid period," which predictive and random
  both do (grid-ablation wanders less, so aliases less).
- It is **not the whole story**: the within-cell spread is still ~0.31 m (~0.4 of a period) and the
  retrained decoder can't recover position — so the other ~half is **genuine within-cell phase
  error**, consistent with the collapse in §1.

## Synthesis
The predictive-ablated decoded "wandering" is **(a) off the torus** — the torus angles are
frozen/clumped while the off-manifold residual drives the output — and the position code genuinely
**collapses** (not a decoder artifact). Of that wandering, **~half is grid-lattice aliasing** (jumps
between grid-field replicas of one phase, exactly Rob's mechanism) and ~half is genuine phase error.
The one **predictive-specific** signature is stalled torus traversal; the *magnitude* of the
wandering/collapse/aliasing is largely a generic consequence of lesioning the integrator, not unique
to PGCs (PGCs carry no unusual weight). Caveat: aliasing is measured against the single dominant grid
module (FFT peak of the grid population); the network has multiple modules whose combination
normally disambiguates absolute position.

*Scripts:* `code/aarav_torus_residual_decomposition.py`, `code/aarav_torus_residual_figure.py`,
`code/aarav_torus_aliasing.py`, `code/aarav_torus_aliasing_figure.py`.
