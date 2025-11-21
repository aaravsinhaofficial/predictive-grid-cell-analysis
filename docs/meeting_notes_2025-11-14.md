# Meeting Notes – 14 Nov 2025

## Agenda recap
- Probe how varying movement speed distributions affects predictive grid scores when trajectories are shifted in time.
- End-of-year goal: demonstrate that predictive grid cells emerge in RNNs and document the effect of removing them.
- Characterize toroidal/continuous-attractor structure that supports the predictive bump once the core analyses are stable.

## Quantitative definitions
- **Predictive grid cells (PGCs)**  
  - Maintain a “normal” grid pattern at zero shift (`gridness₀ ≥ 0.5` for now).  
  - Exhibit a **future-preferring** shift whose peak gridness exceeds the predictive threshold (`gridness_shift ≥ 0.5`) at lags ≥ `min_shift_cm` (default 5 cm).  
  - Optionally require the peak shift to beat a shuffle-derived 95% confidence level (see below).
- **Retrospective grid cells (RGCs)** mirror the PGC definition but prefer lags ≤ `-min_shift_cm`.
- **Zero-lag / classic grid cells** peak within ±`min_shift_cm` while clearing both the zero- and peak-gridness thresholds.
- **Shuffle control** (planned default: 100 permutations per unit) randomizes each unit’s activation trace before recomputing predictive gridness. A unit’s peak lag is significant if the observed gridness exceeds the `(1-α)` percentile (α = 0.05) of the shuffle distribution at the same lag.

## Figure requirements
1. **Counts** – number of predictive vs zero-lag vs retrospective cells at `gridness ≥ 0.5`. Plot as horizontal bars with annotations (`analysis_outputs/*/predictive_retrospective_summary.png`, panel A).
2. **Preferred-shift distribution** – histogram/KDE of the best lag (cm) for PGCs and RGCs (panel B). Overlay the ±`min_shift_cm` decision boundary and report means/medians.
3. **Predictive vs retrospective strength** – side-by-side comparison of zero-shift vs peak-shift gridness for both classes (panel C). This highlights how much extra gridness is gained when projecting forward vs backward in time.

## Immediate next steps
1. Use `predictive_retrospective_summary.py` (see README) on each checkpoint to export the consolidated figure plus a JSON/NPZ summary of counts and preferred shifts.
2. Tune `min_shift_cm`, `gridness` thresholds, and shuffle α to match the experimental dataset (“Japanese paper”) before locking the numbers for the manuscript.
3. Extend the script with the shuffle-based P-value mask (already supported via `--shuffle_trials`) once we finalize how many permutations are practical per checkpoint.
