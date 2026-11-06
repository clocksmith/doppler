"""Export a layerwise diagnostic plot and CSV; no acceptance decision.

Usage: python3 -P plot-relative-prefix.py <metrics.json[.gz]> <position> <output-stem>
"""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

source, position, stem = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
raw = gzip.decompress(source.read_bytes()) if source.suffix == '.gz' else source.read_bytes()
report = json.loads(raw)
assert report['schema'] == 'doppler.upstream-stage-comparison/v2'
rows = [r for r in report['rows'] if position in r['positions']]
layers = sorted(set(r['layer'] for r in rows))
assert layers == list(range(16)), 'this diagnostic requires all layers 0–15'
table = []
for row in rows:
    point = next(p for p in row['positionMetrics'] if p['position'] == position)
    slices = point['streams'] or [point]
    assert all(s['relativeErrorDefined'] for s in slices)
    table.append(dict(layer=row['layer'], role=row['role'], stage=row['stage'], position=position,
                      referenceSliceRms=point['referenceRms'],
                      referenceStreamRmsMin=min(s['referenceRms'] for s in slices),
                      referenceStreamRmsMax=max(s['referenceRms'] for s in slices),
                      maxAbsoluteDifference=point['maxAbsoluteDifference'], rmsDifference=point['rmsDifference'],
                      maxErrorOverReferenceRms=max(s['maxErrorOverReferenceRms'] for s in slices),
                      rmsErrorOverReferenceRms=max(s['rmsErrorOverReferenceRms'] for s in slices)))
with stem.with_suffix('.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(table[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(table)

plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': hashlib.sha256(raw).hexdigest(), 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
colors = dict(input='#1769aa', norm='#d65f16', down='#6750a4', up='#008573', branchInput='#b03c73')
names = dict(input='Residual input', norm='After grouped norm', down='Down projection',
             up='Up projection', branchInput='Mixed branch input')
for row_index, (role, label) in enumerate([('attn_hc', 'Attention HC'), ('mlp_hc', 'MLP HC')]):
    for col, stages in enumerate([['input', 'norm'], ['down', 'up', 'branchInput']]):
        ax = axes[row_index, col]
        for stage in stages:
            values = [next(r['maxErrorOverReferenceRms'] for r in table
                           if (r['layer'], r['role'], r['stage']) == (layer, role, stage)) for layer in layers]
            assert np.isfinite(values).all()
            ax.plot(layers, values, marker='o', markersize=3.5, linewidth=1.7, color=colors[stage], label=names[stage])
        ax.set_title(f'{label}: ' + ('input and normalization' if col == 0 else 'projections and mixing'))
        ax.set_yscale('symlog', linthresh=1e-7)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.2)
        ax.set_xticks(layers)
        ax.legend(fontsize=9, frameon=False, loc='lower right')
        ax.set_ylabel('Max absolute error / reference RMS')
        ax.set_xlabel('Decoder layer')
fig.suptitle(f'Flash-Next · token {position} · relative drift before the first changed expert set', fontsize=14)
fig.text(0.5, 0.015,
         'Maximum over the four streams where applicable; otherwise the full vector. Symlog preserves exact zeros.\n'
         'Identified F32 subgroup candidate; unchanged model gate still fails. This is a diagnostic, not a new tolerance.',
         ha='center', fontsize=9, color='#444444')
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(stem.with_suffix('.png'), dpi=160)
fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
svg = stem.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
plt.close(fig)
receipt = dict(scope='Plot of descriptive relative errors, not model acceptance', normalModelQualified=False,
               metricsSha256=hashlib.sha256(raw).hexdigest(), scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               position=position, layers=layers, rowCount=len(table),
               files={stem.with_suffix(suffix).name: hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
                      for suffix in ['.csv', '.png', '.svg']})
stem.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(position=position, layers=len(layers), output=str(stem))))
