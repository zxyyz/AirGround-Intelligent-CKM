"""Check published summaries against per-stream results; no external data required."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'extensions/full_learning_curves_20261030'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    np.testing.assert_allclose(float(actual), float(expected), rtol=1e-11, atol=1e-12)


def interval(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 10 and np.isfinite(values).all()
    mean, sd = values.mean(), values.std(ddof=1)
    radius = t.ppf(.975, 9) * sd / np.sqrt(10)
    return mean, sd, mean - radius, mean + radius


def key(row):
    return row['variant'], int(row['pair']), float(row['ratio'])


def main():
    flow = rows(EXT / 'results/per_flow_200.csv')
    summary = rows(EXT / 'results/summary_20_cells.csv')
    assert len(flow) == 200 and len(summary) == 20
    lookup = {key(row): row for row in flow}
    assert len(lookup) == 200
    variants = ['Coord', 'RawMap', 'GeoPool', 'GeoDirect', 'Full']
    assert set(lookup) == {(v, p, r) for v in variants
                           for p in range(5101, 5111) for r in [.1, .25, .5, 1.]}
    for row in summary:
        group = [lookup[row['variant'], p, float(row['ratio'])] for p in range(5101, 5111)]
        close(row['n'], len(group))
        for suffix, value in zip(['mean', 'sd', 'ci_low', 'ci_high'],
                                 interval([r['gospa'] for r in group])):
            close(row['gospa_' + suffix], value)
        close(row['n_train_mean'], np.mean([float(r['n_train']) for r in group]))
    original = rows(ROOT / 'results/per_flow.csv')
    assert len(original) == 140
    for row in original:
        for field, value in row.items():
            assert lookup[key(row)][field] == value, (key(row), field)
    for row in rows(EXT / 'results/descriptive_contrasts_by_ratio.csv'):
        ratio = float(row['ratio'])
        treatment = np.array([float(lookup[row['treatment'], p, ratio]['gospa']) for p in range(5101, 5111)])
        control = np.array([float(lookup[row['control'], p, ratio]['gospa']) for p in range(5101, 5111)])
        delta = control - treatment
        relative = 100 * delta / control
        for fields, values in [(['absolute_improvement', 'absolute_sd', 'ci_low', 'ci_high'], interval(delta)),
                               (['relative_improvement_percent', None, 'relative_ci_low', 'relative_ci_high'], interval(relative))]:
            for field, value in zip(fields, values):
                if field:
                    close(row[field], value)
        close(row['favorable_pairs'], (delta > 0).sum())
    primary = rows(ROOT / 'results/primary_contrasts.csv')
    assert len(primary) == 4
    probabilities = []
    for row in primary:
        a = np.array([float(lookup[row['treatment'], p, 1.]['gospa']) for p in range(5101, 5111)])
        b = np.array([float(lookup[row['control'], p, 1.]['gospa']) for p in range(5101, 5111)])
        mean, sd, lo, hi = interval(b - a)
        rel, _, rlo, rhi = interval(100 * (b - a) / b)
        for field, value in [('mean_improvement', mean), ('sd', sd), ('ci_low', lo), ('ci_high', hi),
                             ('mean_relative_improvement_percent', rel), ('relative_ci_low', rlo), ('relative_ci_high', rhi)]:
            close(row[field], value)
        probability = ttest_1samp(b - a, 0).pvalue
        close(row['p'], probability)
        probabilities.append(probability)
    previous = 0.
    for rank, index in enumerate(np.argsort(probabilities)):
        previous = max(previous, min(1., (4 - rank) * probabilities[index]))
        close(primary[index]['p_holm'], previous)
    plotted = rows(ROOT / 'figures/plotted_data.csv')
    assert len(plotted) == 20
    for row in plotted:
        record = next(s for s in summary if s['variant'] == row['variant'] and float(s['ratio']) == float(row['ratio']))
        for plot_field, summary_field in [('x', 'n_train_mean'), ('mean', 'gospa_mean'), ('sd', 'gospa_sd')]:
            close(row[plot_field], record[summary_field])
    scale = rows(ROOT / 'results/scale_sensitivity_per_flow.csv')
    scales = {r['scale'] for r in scale}
    assert len(scales) == 15 and len(scale) == 750
    scaled = {(r['scale'], r['variant'], int(r['pair'])): float(r['gospa']) for r in scale}
    assert len(scaled) == 750
    contrasts = rows(ROOT / 'results/scale_sensitivity_contrasts.csv')
    assert len(contrasts) == 60
    for row in contrasts:
        a = np.array([scaled[row['scale'], row['treatment'], p] for p in range(5101, 5111)])
        b = np.array([scaled[row['scale'], row['control'], p] for p in range(5101, 5111)])
        mean, _, lo, hi = interval(b - a)
        rel, _, rlo, rhi = interval(100 * (b - a) / b)
        for field, value in [('mean_improvement', mean), ('ci_low', lo), ('ci_high', hi),
                              ('mean_relative_improvement_percent', rel), ('relative_ci_low', rlo), ('relative_ci_high', rhi)]:
            close(row[field], value)
        if row['treatment'] == 'Full':
            assert mean > 0
    freeze = json.loads((ROOT / 'configs/freeze.json').read_text())
    for name, expected in freeze['training_source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print('PASS: 200 runs, 20 cells, original 140 rows unchanged, paired contrasts, 15 scales, frozen scientific source.')


if __name__ == '__main__':
    main()
