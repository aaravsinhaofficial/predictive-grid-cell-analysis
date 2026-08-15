"""Aggregate the doors-fleet blitz into the ecology verdict.

Reads each arm's blitz suite_summary.json (+ training metrics for context)
and prints/writes the decision table:
  reliance   = sham - all_ro   (does the doors task make the code load-bearing?)
  predictive = sham - predictive vs the matched/random controls
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BANINO = os.path.expanduser(os.environ.get('BANINO_REPO', '~/banino'))
ARMS = ['pirnn_doors_s0', 'pirnn_doors_s2', 'pirnn_doors_s20',
        'pirnn_doors_s0_lr2', 'pirnn_doors_s0_aws']


def latest_frames(run):
  try:
    rows = [json.loads(l) for l in
            open(os.path.join(BANINO, 'rl_runs', run, 'metrics.jsonl'))]
    return rows[-1]['frames'] / 1e6, rows[-1]['avg_return_50'], \
        rows[-1]['ent']
  except Exception:
    return float('nan'), float('nan'), float('nan')


def main():
  out = {}
  pooled = {}
  for arm in ARMS:
    p = os.path.join(BANINO, 'rl_runs', arm, 'ablation_evals', 'blitz',
                     'suite_summary.json')
    fr, ret, ent = latest_frames(arm)
    if not os.path.exists(p):
      print(f'{arm:22s} trained {fr:6.1f}M ret {ret:5.1f} ent {ent:.3f}  '
            f'[no blitz summary]')
      continue
    s = json.load(open(p))
    row = dict(frames_M=round(fr, 1), train_ret=ret, ent=ent,
               sham=[s['sham_mean'], s['sham_sem']])
    for k in ('chance', 'all_ro', 'predictive', 'matched', 'randgrid_0'):
      c = s['conditions'].get(k)
      if c:
        row[k] = [c['mean'], c['sem'], c['delta_vs_sham'], c['welch_p']]
        pooled.setdefault(k, []).append(
            (c['mean'], c['sem'], s['sham_mean'], s['sham_sem'],
             np.array(json.load(open(os.path.join(
                 BANINO, 'rl_runs', arm, 'ablation_evals', 'blitz',
                 k + '.json')))['scores'])))
    out[arm] = row
    ch = row.get('chance', [float('nan')] * 4)[0]
    ar = row.get('all_ro', [float('nan')] * 4)
    pr = row.get('predictive', [float('nan')] * 4)
    print(f"{arm:22s} {fr:6.1f}M  sham {row['sham'][0]:5.2f}±"
          f"{row['sham'][1]:4.2f}  chance {ch:5.2f}  "
          f"all_ro Δ{ar[2]:+6.2f} (p={ar[3]:.3f})  "
          f"pred Δ{pr[2]:+6.2f} (p={pr[3]:.3f})")

  # Pooled across-arm test (per-episode scores concatenated, arm-centred).
  if out:
    from scipy import stats
    pooled_tests = {}
    sham_all = []
    for arm, row in out.items():
      f = os.path.join(BANINO, 'rl_runs', arm, 'ablation_evals', 'blitz',
                       'sham.json')
      if os.path.exists(f):
        sham_all.append((arm, np.array(json.load(open(f))['scores'])))
    for k in ('all_ro', 'predictive', 'matched', 'randgrid_0'):
      ds = []
      for arm, srow in sham_all:
        f = os.path.join(BANINO, 'rl_runs', arm, 'ablation_evals', 'blitz',
                         k + '.json')
        if not os.path.exists(f):
          continue
        cond = np.array(json.load(open(f))['scores'])
        ds.append(cond.mean() - srow.mean())
      if len(ds) >= 3:
        t, p = stats.ttest_1samp(ds, 0.0)
        pooled_tests[k] = dict(mean_delta=float(np.mean(ds)),
                               per_arm=[float(d) for d in ds],
                               t=float(t), p=float(p), n_arms=len(ds))
        print(f'POOLED {k:12s} mean Δ {np.mean(ds):+6.2f} across '
              f'{len(ds)} arms (t={t:.2f}, p={p:.3f})')
    out['_pooled'] = pooled_tests
  with open(os.path.join(HERE, 'results', 'doors_verdict.json'), 'w') as f:
    json.dump(out, f, indent=1)
  print('wrote results/doors_verdict.json')


if __name__ == '__main__':
  main()
