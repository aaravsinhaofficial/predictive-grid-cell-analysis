"""Figures for the RL navigation ablation experiment.

Reads results/*.json + *_metrics.jsonl (run `run_ablation_suite.py` and the
collection step first) and writes results/rl_navigation_results.png.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

# Validated categorical palette (dataviz reference, light mode, fixed order).
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
     '#4a3aa7', '#e34948']
GRAY, LGRAY = '#4a4a45', '#b9b8b0'

RUNS = [  # (metrics stem, display name, palette slot)
    ('pirnn_fake_s0', 'FakeEnv s0 (sighted)', 0),
    ('pirnn_fake_s2', 'FakeEnv s2 (sighted)', 1),
    ('pirnn_fake_s20', 'FakeEnv s20 (sighted)', 2),
    ('pirnn_fake_s0_blind', 'FakeEnv s0 (blind)', 3),
    ('pirnn_fake_s2_blind', 'FakeEnv s2 (blind)', 4),
    ('pirnn_arena_s0', 'DM-Lab arena s0', 6),
]
SUITES = [  # (summary stem, display name)
    ('pirnn_fake_s0_blind_final', 'blind s0\n(106M)'),
    ('pirnn_fake_s2_blind_final', 'blind s2\n(112M)'),
    ('pirnn_fake_s0_final', 'sighted s0\n(304M)'),
    ('pirnn_arena_s0_final', 'arena s0\n(192M)'),
]
CONDS = [  # (key, label, color)
    ('sham', 'sham', GRAY),
    ('predictive', 'predictive', C[0]),
    ('matched', 'matched ctrl', C[1]),
    ('randgrid', 'random grid (5x)', C[2]),
    ('zerolag_top', 'zero-lag top', C[3]),
    ('retrospective', 'retrospective', C[4]),
    ('all_ro', 'code fully masked', C[6]),
    ('chance', 'untrained', LGRAY),
]


def load(stem):
  p = os.path.join(RES, stem + '.json')
  return json.load(open(p)) if os.path.exists(p) else None


def cond_stats(s, key):
  if key == 'sham':
    return s['sham_mean'], s['sham_sem']
  if key == 'randgrid':
    ms = [v for k, v in s['conditions'].items()
          if k.startswith('randgrid') and not k.endswith('_ro')]
    return (float(np.mean([m['mean'] for m in ms])),
            float(np.mean([m['sem'] for m in ms])))
  c = s['conditions'].get(key)
  return (c['mean'], c['sem']) if c else (np.nan, np.nan)


def smooth(x, k=9):
  if len(x) < k:
    return np.asarray(x, float)
  return np.convolve(x, np.ones(k) / k, mode='valid')


fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), dpi=160)
fig.patch.set_facecolor('white')
for ax in axes.flat:
  ax.set_facecolor('white')
  for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
  ax.grid(axis='y', color='#eceae4', lw=0.8)
  ax.set_axisbelow(True)
  ax.tick_params(colors=GRAY, labelsize=8)

# --- A: learning curves -------------------------------------------------
ax = axes[0, 0]
for stem, name, slot in RUNS:
  p = os.path.join(RES, stem + '_metrics.jsonl')
  if not os.path.exists(p):
    continue
  rows = [json.loads(l) for l in open(p)]
  f = np.array([r['frames'] for r in rows]) / 1e6
  y = np.array([r['avg_return_50'] for r in rows])
  ys = smooth(y)
  fs = f[len(f) - len(ys):]
  ax.plot(fs, ys, lw=2, color=C[slot], label=name)
  ax.annotate(name, (fs[-1], ys[-1]), textcoords='offset points',
              xytext=(4, 0), fontsize=7, color=C[slot], va='center')
ax.axhline(4.5, color=LGRAY, lw=1.5, ls='--')
ax.annotate('untrained chance', (2, 4.9), fontsize=7, color=GRAY)
ax.set_xlim(0, 390)
ax.set_xlabel('environment frames (M)', fontsize=9, color=GRAY)
ax.set_ylabel('episode return (50-ep avg)', fontsize=9, color=GRAY)
ax.set_title('A  Agents on the frozen PI-RNN code learn the goal task',
             fontsize=10, loc='left', color='#1a1a19')
ax.legend(fontsize=7, frameon=False, loc='upper left')

# --- B: ablation scores -------------------------------------------------
ax = axes[0, 1]
ng = len(SUITES)
nc = len(CONDS)
w = 0.8 / nc
for ci, (key, label, col) in enumerate(CONDS):
  xs, ms, es = [], [], []
  for gi, (stem, gname) in enumerate(SUITES):
    s = load(stem)
    if not s:
      continue
    m, e = cond_stats(s, key)
    xs.append(gi + (ci - nc / 2 + 0.5) * w)
    ms.append(m)
    es.append(e)
  ax.bar(xs, ms, width=w * 0.92, color=col, label=label, zorder=3)
  ax.errorbar(xs, ms, yerr=es, fmt='none', ecolor=GRAY, lw=1, capsize=0,
              zorder=4)
ax.set_xticks(range(ng))
ax.set_xticklabels([g for _, g in SUITES], fontsize=8, color=GRAY)
ax.set_ylabel('score (mean episode return)', fontsize=9, color=GRAY)
ax.set_title('B  No lesion (incl. removing the whole code) hurts trained '
             'navigation', fontsize=10, loc='left', color='#1a1a19')
ax.legend(fontsize=7, frameon=False, ncol=2)

# --- C: mechanism (decode error) ---------------------------------------
ax = axes[1, 0]
MCONDS = [('sham', 'sham', GRAY), ('predictive', 'predictive', C[0]),
          ('matched', 'matched ctrl', C[1]),
          ('zerolag_top', 'zero-lag top', C[3])]
w = 0.8 / len(MCONDS)
for ci, (key, label, col) in enumerate(MCONDS):
  xs, ds = [], []
  for gi, (stem, gname) in enumerate(SUITES):
    s = load(stem)
    if not s:
      continue
    d = (np.mean([v['decode_err_cm'] for k, v in s['conditions'].items()
                  if k == key]) if key != 'sham'
         else min(v['decode_err_cm'] for v in s['conditions'].values()))
    xs.append(gi + (ci - len(MCONDS) / 2 + 0.5) * w)
    ds.append(d)
  ax.bar(xs, ds, width=w * 0.92, color=col, label=label, zorder=3)
ax.set_xticks(range(ng))
ax.set_xticklabels([g for _, g in SUITES], fontsize=8, color=GRAY)
ax.set_ylabel('place-cell decode error (cm)', fontsize=9, color=GRAY)
ax.set_title('C  The same lesions DO degrade the code (mechanism intact)',
             fontsize=10, loc='left', color='#1a1a19')
ax.legend(fontsize=7, frameon=False)

# --- D: forest plot — predictive-ablation effect, every evaluation ------
ax = axes[1, 1]
EVALS = [  # (stem, label, palette slot)
    ('pirnn_fake_s0_blind_final', 'blind s0, 106M (n=150)', 3),
    ('pirnn_fake_s2_blind_final', 'blind s2, 112M (n=150)', 4),
    ('pirnn_fake_s0_final', 'sighted s0, 304M (n=100)', 0),
    ('pirnn_arena_s0_final', 'arena s0, 192M (n=80)', 6),
    ('pirnn_arena_s0_mid128M', 'arena s0, 128M (n=80)', 6),
    ('pirnn_fake_s0_blind_mid67M', 'blind s0, 67M (n=100)', 3),
    ('pirnn_fake_s0_blind_mid67M_n400', 'blind s0, 67M (n=400 confirm)', 3),
    ('pirnn_fake_s2_blind_mid78M', 'blind s2, 78M (n=100 replicate)', 4),
]
ys, labels = [], []
for i, (stem, label, slot) in enumerate(EVALS):
  s = load(stem)
  if not s:
    continue
  c = s['conditions'].get('predictive')
  if not c:
    continue
  d = c['delta_vs_sham']
  ci = 1.96 * np.hypot(c['sem'], s['sham_sem'])
  y = len(ys)
  ax.errorbar([d], [y], xerr=[ci], fmt='o', ms=6, color=C[slot],
              ecolor=C[slot], elinewidth=2, capsize=0, zorder=3)
  ys.append(y)
  labels.append(label)
ax.axvline(0, color=GRAY, lw=1.2)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=7.5, color=GRAY)
ax.invert_yaxis()
ax.set_xlabel('predictive-ablation effect on score, Δ vs sham (95% CI)',
              fontsize=9, color=GRAY)
ax.set_title('D  Predictive ablation: no behavioural deficit in any '
             'evaluation', fontsize=10, loc='left', color='#1a1a19')
ax.grid(axis='x', color='#eceae4', lw=0.8)
ax.grid(axis='y', visible=False)

fig.tight_layout(pad=1.6)
out = os.path.join(RES, 'rl_navigation_results.png')
fig.savefig(out, bbox_inches='tight')
print('wrote', out)
