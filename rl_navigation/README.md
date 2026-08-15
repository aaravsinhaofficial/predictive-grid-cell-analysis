# RL navigation test of predictive grid cells

**Question.** Do the predictive grid cells identified in the cohort RNNs
matter for goal-directed navigation, not just for path-integration decoding?
The matched-ablation study showed predictive/retrospective units are
selectively important for *decoding* (slope 0.139 vs matched 0.045 cm/unit),
but the random-walk path-dependence effect is modest. A behavioural
navigation deficit after predictive-cell ablation would be stronger evidence
of function.

**Design.** Within-agent, eval-time lesion. We train an RL agent whose only
spatial code is the frozen hidden state of a cohort path-integration RNN,
then lesion unit groups at evaluation and measure the drop in navigation
score. Training is done once per RNN seed; every condition evaluates the
same policy weights, so any score difference is attributable to the lesion.

## Agent (banino repo, `rl/`, `--agent pirnn`; commit 508c849)

The Banino-2018 reproduction framework's agent is modular: a policy LSTM
consumes `[conv features, grid code, goal code, prev action, prev reward]`,
where the goal code is a snapshot of the grid code at the last reward. We
replace the framework's separately-trained GridModule with the **frozen**
cohort RNN (`Models/canonical_cohort_v1/seed_N/.../most_recent_model.pth`):

- The policy receives the raw 4096-d RNN hidden state `h` as the grid code
  (and hence 4096-d goal codes). The RNN gets **no gradients** and is never
  fine-tuned.
- Input at each decision step is the allocentric displacement
  `Δpos · (1.1 / env_half)` — the RNN's native convention (metres per step
  in its [-1.1, 1.1] m box). FakeEnv steps (≤0.023 m·0.8) sit inside the
  training displacement distribution (mean 0.020 m).
- At episode start and at goal-respawn teleports, `h` is re-derived through
  the RNN's own init path: `h₀ = encoder(place-cell activation at pos)`,
  followed by one zero-velocity recurrence step. The same re-anchoring is
  applied with p = 0.05 per step, mirroring the 5% visual-correction input
  of the paper agent's grid LSTM — a pure integrator drifts ~50 cm by 300
  steps, which would teach the policy to distrust the code (`rl/test_pi_rnn.py`
  measures the drift).
- `rl/test_pi_rnn.py` verifies: stepwise recurrence is bit-exact against
  `model.g`, place-cell centres/activations match, and 2000-step rollouts
  stay finite.

## Lesion semantics (`rl/eval_pirnn.py`)

- `full` — the paper pipeline's `zero_unit_weights_in_place`, verified
  bit-exact: zero encoder row, `W_ih` row, `W_hh` row + column, decoder
  column. The unit is dead in the dynamics, in re-anchoring, and in what the
  policy sees.
- `readout` — dynamics left intact; only the code entering the policy (and
  the goal snapshot) is masked. Separates "the code degraded" from "the
  policy lost these input features".

## Conditions (`run_ablation_suite.py`, per seed)

Sham; predictive (all `labels==1` units); retrospective; count-matched
strongest zero-lag units; **property-matched control** (the pipeline's
`select_matched_control`, greedy NN in the 8-covariate z-space over the
grid pool minus predictive); 5× count-matched random grid-pool draws;
5× count-matched random draws from all evaluated units; plus readout-mode
predictive/matched/random and a full-code mask (`all_ro`, reliance ceiling).
Score = mean episode return (SI protocol, stochastic policy); each
condition also reports the lesioned RNN's place-cell decode error on the
behaved trajectories (mechanism check).

## Run matrix (launched 2026-08-15, local, 2× RTX PRO 6000)

| run | env | RNN seed | why this seed |
|---|---|---|---|
| `pirnn_fake_s0` | FakeEnv | 0 | 33 predictive units; "real-gap" seed |
| `pirnn_fake_s2` | FakeEnv | 2 | 30 predictive; real-gap, high decode slope |
| `pirnn_fake_s20` | FakeEnv | 20 | strongest per-unit decode slope (0.936) |
| `a3c_fake` | FakeEnv | — | no-spatial-code baseline (framework `a3c`) |
| `pirnn_arena_s0` | DM-Lab `square_arena_goal` | 0 | flagship, paper's actual task |

FakeEnv is the framework's LabEnv-contract goal task (hidden goal, +10 and
respawn on reach, 1800-step episodes, motor noise): the same
find-then-return structure as the DM-Lab arena at ~1.7× the frame rate and
with no visual confounds. 16 envs, 400M-frame budget, SI Table 2
hyperparameters (see `launch_runs.sh` for exact commands).

## Interpretation guide

- Navigation damage that is predictive > matched control (and > random) at
  similar lesion counts = the paper's decoding-selectivity result, upgraded
  to behaviour.
- predictive ≈ matched > sham-level damage = grid-cell importance without
  predictive-specific function.
- Nothing beats sham while `all_ro` does = the policy uses the code but any
  61-unit lesion is below the behavioural detection floor.
- `all_ro` ≈ sham = the policy ignored the code (navigates by vision);
  compare against `a3c_fake` before concluding anything.

## Known deviations / caveats

- Allocentric ground-truth displacement input (the Sorscher RNN's native
  format) and place-cell re-anchoring both use privileged position — as did
  the framework's replay-supervised grid trainers, but here at acting time.
  The experiment tests the *code*, not sensory-driven localization.
- Only units 0–255 of 4096 were ever classified by the PGC pipeline; all
  lesion groups live in that subspace, and 3840 unclassified units remain
  intact in every condition.
- The RNN was trained on 20-step random-walk sequences in a 2.2 m box;
  RL trajectories are longer-horizon and goal-directed (more predictable —
  where predictive coding should matter most).
