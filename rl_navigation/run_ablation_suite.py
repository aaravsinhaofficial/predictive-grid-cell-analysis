"""Eval-time PGC ablation suite for the RL navigation experiment.

For one trained ``--agent pirnn`` run (banino rl/train_rl.py) whose grid code
is a frozen canonical-cohort path-integration RNN, evaluates navigation score
under a battery of unit lesions and controls:

  full mode (source-repo lesion semantics, zero_unit_weights_in_place):
    sham            no ablation
    predictive      all predictive units (labels==1)
    retrospective   all retrospective units (labels==2)
    zerolag_top     count-matched strongest zero-lag grid units (labels==0)
    matched         property-matched control (greedy NN in the 8-covariate
                    z-space, pool = grid-union minus predictive, exactly the
                    paper pipeline's select_matched_control)
    randgrid_k      count-matched random draws from the same pool (k trials)
    rand256_k       count-matched random draws from all evaluated units
                    excluding predictive (k trials)
  readout mode (policy-input deprivation only; RNN dynamics intact):
    predictive_ro, matched_ro, randgrid_ro_k, all_ro (mask entire code)

Scores come from rl/eval_pirnn.py (SI protocol: stochastic policy, N-episode
mean score); each condition also logs the on-trajectory place-cell decode
error of the lesioned RNN as the mechanism-level readout.

Usage:
  python3 rl_navigation/run_ablation_suite.py \
      --run ~/banino/rl_runs/pirnn_fake_s0 --seed 0 --episodes 100
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

BANINO = os.path.expanduser(os.environ.get('BANINO_REPO',
                                           '~/banino'))
PGC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_NAME = ('steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_'
            'False_lr_00001_weight_decay_1e-06')
COVARIATE_KEYS = ('module', 'gridness', 'bandness', 'rate_var', 'hd_r',
                  'decoder_mag', 'rec_in', 'rec_out')  # pgc_common order


def load_paper_helpers():
  """Extract the numpy-only matching helpers from pgc_matched_ablation.py
  without importing it (its import chain needs cv2/imageio)."""
  src = open(os.path.join(PGC, 'code', 'pgc_matched_ablation.py')).read()
  ns = {'np': np}
  for fn in ('rank_within_class', '_standardize_over_pool',
             'select_matched_control'):
    m = re.search(rf'\ndef {fn}.*?(?=\n(?:def |# ---))', src, re.DOTALL)
    exec(m.group(0), ns)  # noqa: S102 - trusted local repo source
  return ns


def unit_sets(seed, rng, random_trials):
  d = os.path.join(PGC, 'analysis_outputs', 'canonical_cohort_v1',
                   f'seed_{seed}', RUN_NAME, 'pgc_rigor')
  z = np.load(os.path.join(d, 'pgc_classification.npz'))
  cov = np.load(os.path.join(d, 'pgc_covariates.npz'))
  labels = z['labels'].astype(int)
  P = np.stack([cov[k] for k in COVARIATE_KEYS], axis=1)
  H = load_paper_helpers()

  pred = np.where(labels == 1)[0]
  retro = np.where(labels == 2)[0]
  zerolag = np.where(labels == 0)[0]
  grid_union = np.unique(np.concatenate([pred, retro, zerolag]))
  pool = np.setdiff1d(grid_union, pred)
  n_p = pred.size

  sets = {'predictive': pred, 'retrospective': retro}
  zl_ranked = H['rank_within_class'](zerolag, z['best_cm'],
                                     z['best_gridness'])
  sets['zerolag_top'] = np.sort(zl_ranked[:n_p])
  sets['matched'] = np.sort(H['select_matched_control'](pred, P, pool, rng))
  for k in range(random_trials):
    sets[f'randgrid_{k}'] = np.sort(rng.choice(pool, min(n_p, pool.size),
                                               replace=False))
    non_pred = np.setdiff1d(np.arange(labels.size), pred)
    sets[f'rand256_{k}'] = np.sort(rng.choice(non_pred, n_p, replace=False))
  meta = dict(seed=seed, n_predictive=int(n_p),
              n_retrospective=int(retro.size), n_zerolag=int(zerolag.size),
              grid_pool_size=int(pool.size))
  return sets, meta


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run', required=True)
  ap.add_argument('--seed', type=int, required=True,
                  help='cohort seed of the RNN inside this run')
  ap.add_argument('--ckpt', default='latest')
  ap.add_argument('--episodes', type=int, default=100)
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--random_trials', type=int, default=5)
  ap.add_argument('--match_seed', type=int, default=1234)
  ap.add_argument('--skip_readout', action='store_true')
  ap.add_argument('--label', default=None,
                  help='subdir label for this eval batch (default: ckpt id)')
  args = ap.parse_args()
  run = os.path.abspath(os.path.expanduser(args.run))

  rng = np.random.default_rng(args.match_seed + args.seed)
  sets, meta = unit_sets(args.seed, rng, args.random_trials)
  ng = 4096

  # Resolve checkpoint once so every condition scores the same weights.
  if args.ckpt in ('latest', 'final'):
    import glob
    cands = (sorted(glob.glob(os.path.join(run, 'ckpt_*.pt'))) or
             [os.path.join(run, 'ckpt_final.pt')])
    ckpt = cands[-1]
  else:
    ckpt = args.ckpt
  label = args.label or os.path.basename(ckpt).replace('.pt', '')
  out_dir = os.path.join(run, 'ablation_evals', label)
  os.makedirs(out_dir, exist_ok=True)

  conditions = [('sham', None, 'full'), ('chance', None, 'full')]
  for name, units in sets.items():
    conditions.append((name, units, 'full'))
  if not args.skip_readout:
    for name in ('predictive', 'matched', 'randgrid_0', 'randgrid_1'):
      conditions.append((f'{name}_ro', sets[name], 'readout'))
    conditions.append(('all_ro', np.arange(ng), 'readout'))

  results = {}
  for tag, units, mode in conditions:
    out_json = os.path.join(out_dir, f'{tag}.json')
    if os.path.exists(out_json):
      with open(out_json) as f:
        results[tag] = json.load(f)
      print(f'[suite] {tag}: cached ({results[tag]["mean_score"]:.2f})',
            flush=True)
      continue
    cmd = [sys.executable, '-m', 'rl.eval_pirnn', '--run', run,
           '--ckpt', ckpt, '--episodes', str(args.episodes),
           '--n_envs', str(args.n_envs), '--device', args.device,
           '--tag', tag, '--ablation_mode', mode, '--out', out_json]
    if tag == 'chance':
      cmd += ['--random_policy']
    if units is not None:
      uf = os.path.join(out_dir, f'{tag}_units.json')
      with open(uf, 'w') as f:
        json.dump([int(u) for u in units], f)
      cmd += ['--ablate', '@' + uf]
    print(f'[suite] running {tag} ({mode}, '
          f'n={0 if units is None else len(units)})...', flush=True)
    subprocess.run(cmd, cwd=BANINO, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(out_json) as f:
      results[tag] = json.load(f)
    print(f'[suite] {tag}: score {results[tag]["mean_score"]:.2f} '
          f'± {results[tag]["sem"]:.2f}, '
          f'decode {results[tag]["decode_err_cm"]:.1f} cm', flush=True)

  # ---- aggregate: score deltas + tests vs sham ----
  from scipy import stats
  sham = np.array(results['sham']['scores'])
  summary = dict(meta=meta, ckpt=ckpt, episodes=args.episodes,
                 sham_mean=float(sham.mean()),
                 sham_sem=float(sham.std() / np.sqrt(sham.size)),
                 conditions={})
  for tag, r in results.items():
    if tag == 'sham':
      continue
    s = np.array(r['scores'])
    t, p_t = stats.ttest_ind(s, sham, equal_var=False)
    try:
      u, p_u = stats.mannwhitneyu(s, sham, alternative='two-sided')
    except ValueError:
      p_u = float('nan')
    summary['conditions'][tag] = dict(
        mean=float(s.mean()), sem=float(s.std() / np.sqrt(s.size)),
        delta_vs_sham=float(s.mean() - sham.mean()),
        pct_of_sham=float(100 * s.mean() / sham.mean()) if sham.mean() else None,
        welch_p=float(p_t), mannwhitney_p=float(p_u),
        n_ablated=r['n_ablated'], mode=r['ablation_mode'],
        decode_err_cm=r['decode_err_cm'])
  with open(os.path.join(out_dir, 'suite_summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
  print(json.dumps(summary, indent=1))


if __name__ == '__main__':
  main()
