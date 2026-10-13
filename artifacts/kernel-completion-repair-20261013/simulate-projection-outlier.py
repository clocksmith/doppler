"""CPU diagnostic of retained operands; no model reference or GPU work changes.

Hypothesis: rounding in the declared two-component reduction reproduces the
physical outlier without a provider arithmetic defect. Bitwise disagreement
between simulated and retained pairs would disprove that explanation.
"""
from pathlib import Path
from fractions import Fraction as Q
import json, struct, hashlib
ROOT = Path(__file__).resolve().parent
raw = (ROOT / 'projection-outlier-result.json').read_bytes()
receipt = json.loads(raw)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def normalize(high, low):
    rounded = f32(high + low)
    return [rounded, f32(Q(high) + Q(low) - Q(rounded))]


def add(a, b):
    high = f32(a[0] + b[0])
    error = f32(Q(a[0]) + Q(b[0]) - Q(high))
    low = f32(a[1] + b[1])
    joined = f32(error + low)
    tail = f32(f32(Q(a[1]) + Q(b[1]) - Q(low))
               + f32(Q(error) + Q(low) - Q(joined)))
    result = normalize(high, joined)
    return normalize(result[0], f32(result[1] + tail))


def multiply_scalar(a, b):
    high, low = f32(a[0] * b), f32(a[1] * b)
    return add([high, f32(Q(a[0]) * Q(b) - Q(high))],
               [low, f32(Q(a[1]) * Q(b) - Q(low))])


results = []
for row in receipt['outliers']:
    # These saved finite normal operands use exact rational residuals; this is
    # not an emulator for the shader's general subnormal/overflow behavior.
    exact_dot = sum(((Q(row['input'][2*k]) + Q(row['input'][2*k+1])) * Q(w)
                     for k, w in enumerate(row['weights'])), Q(0))
    partials = []
    exact_lanes = []
    for lane in range(64):
        total = [0.0, 0.0]
        exact_lane = Q(0)
        for k in range(lane, row['K'], 64):
            a, weight = row['input'][k*2:k*2+2], row['weights'][k]
            total = add(total, multiply_scalar(a, weight))
            exact_lane += (Q(a[0]) + Q(a[1])) * Q(weight)
        partials.append(total)
        exact_lanes.append(exact_lane)
    lane_error = sum((Q(p[0]) + Q(p[1]) - e for p, e in zip(partials, exact_lanes)), Q(0))
    for stride in [32, 16, 8, 4, 2, 1]:
        for lane in range(stride):
            partials[lane] = add(partials[lane], partials[lane + stride])
    high = f32(row['alpha'])
    low = f32(row['alpha'] - high)
    result = add(multiply_scalar(partials[0], high), multiply_scalar(partials[0], low))
    exact = exact_dot * Q(row['alpha'])
    error = Q(result[0]) + Q(result[1]) - exact
    results.append({
        'row': row['row'], 'column': row['col'], 'simulatedPair': result,
        'retainedGpuPair': row['actualPair'],
        'bitsMatch': struct.pack('<ff', *result) == struct.pack('<ff', *row['actualPair']),
        'summedLaneAbsoluteError': float(abs(lane_error)),
        'reducedDotAbsoluteError': float(abs(Q(partials[0][0]) + Q(partials[0][1]) - exact_dot)),
        'relativeErrorAfterAlpha': float(abs(error) / max(Q(1), abs(exact))),
        'originalThresholdPassed': abs(error) / max(Q(1), abs(exact)) <= Q(receipt['tolerance']),
    })
report = {
    'schema': 'doppler.projection-outlier-cpu-simulation/v1',
    'sourceSha256': hashlib.sha256(raw).hexdigest(),
    'hypothesis': 'Two-component rounding reproduces the retained physical outlier',
    'scope': 'One retained synthetic dot product; not general shader or model qualification',
    'originalTolerance': receipt['tolerance'], 'results': results,
    'runtimeChanged': False, 'approvedReferenceChanged': False,
}
(ROOT / 'projection-outlier-simulation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
assert all(row['bitsMatch'] for row in results)
