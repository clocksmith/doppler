import collections
import hashlib
import json
import math
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent
read = lambda name: json.loads((root / name).read_text())
probe = read('binding-attribution-receipt.json')
journey = read('baseline-journey-receipt.json')
contract = read('probe-contract.json')
successor = read('successor-verification.json')
js = read('successor-js-verification.json')
for receipt in [probe, journey]:
    assert receipt['passed']
    assert receipt['archiveSha256'] == contract['baselineArchiveSha256']
    assert receipt['modelManifestSha256'] == contract['manifestSha256']
    assert receipt['configSha256'] == contract['configSha256']
assert successor['passed'] and js['passed']
assert successor['successorManifestSha256'] == contract['manifestSha256']
assert js['manifestSha256'] == contract['manifestSha256']
for filename, receipt in [('binding-attribution.js', probe), ('baseline-journey.js', journey)]:
    assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == receipt['driverSha256']
decode = {k: v for k, v in probe['result']['bindingRows'].items() if k.startswith('batch_decode/')}
operations = collections.defaultdict(collections.Counter)
changes = collections.defaultdict(collections.Counter)
for label, values in decode.items():
    operation = label.split(':L')[0]
    operations[operation].update({k: v for k, v in values.items() if isinstance(v, int)})
    for binding, counts in values.get('bindingChanges', {}).items():
        changes[operation].update({f'binding{binding}.{key}': value for key, value in counts.items()})
rows = journey['result']['rows']
warm = [row for row in rows if row['stage'] == 'warm']
percentile = lambda values, p: sorted(values)[max(0, math.ceil(len(values) * p) - 1)]
gaps = [b - a for row in warm for a, b in zip(row['delivery'], row['delivery'][1:])]
pool_snapshots = [row.get('pool') or row.get('memory', {}).get('pool') for row in rows]
pool_snapshots = [row for row in pool_snapshots if row and 'peakBytesAllocated' in row]
summary = {
    'schema': 'doppler.bind-group-observation/v1',
    'baselineCommit': contract['baselineCommit'],
    'baselineArchiveSha256': contract['baselineArchiveSha256'],
    'successorManifestSha256': contract['manifestSha256'],
    'browser': journey['browser'], 'gpu': journey['hardware']['gpu']['devices'],
    'scope': 'Qwen3.5-2B Q4_K/F32 compute/F16 KV, Chrome Metal on Apple M3, context 2048, greedy 32 output tokens; instrumented binding observations are separate from baseline lifecycle timings.',
    'decision': 'reject binding-only cache for the observed matmul path before implementation',
    'runtimeCandidateImplemented': False,
    'candidatePerformanceMeasured': False,
    'reason': 'No same-recorder binding identity repeats. Input and output GPUBuffer identities change on every successive per-layer decode matmul. Stabilizing scratch allocation is a separate experiment.',
    'recordedDecodeMatmulCalls': sum(row['calls'] for row in decode.values()),
    'sameRecorderIdentityRepeats': sum(row['sameRecorderHits'] for row in decode.values()),
    'lifetimeIdentityRepeats': sum(row['lifetimeHits'] for row in decode.values()),
    'operations': dict(operations), 'bindingChanges': dict(changes),
    'baselineObservations': {
        'openingMs': next(row['openingMs'] for row in rows if row['stage'] == 'open'),
        'warmSamples': len(warm),
        'warmFirstOutputMedianMs': statistics.median(row['firstTokenMs'] for row in warm),
        'warmCompleteMedianMs': statistics.median(row['completeMs'] for row in warm),
        'warmDecodeMedianTokensPerSecond': statistics.median(row['stats']['decodeTokens'] * 1000 / row['stats']['decodeTimeMs'] for row in warm),
        'warmRecordingMedianMs': statistics.median(row['stats']['decodeRecordMs'] for row in warm),
        'warmChunkGapP95Ms': percentile(gaps, .95),
        'longPrefillFirstOutputMs': next(row['firstTokenMs'] for row in rows if row['stage'] == 'long-prefill'),
        'peakPoolAllocatedBytes': max(row['peakBytesAllocated'] for row in pool_snapshots),
        'peakBrowserProcessRssSumBytes': journey['peakRssSumBytes'],
        'caveats': ['three warm samples, no comparative performance claim',
            'fresh Chrome process and HTTP load; OS disk cache was not flushed',
            'nonexclusive host; unrelated contributor processes were not modified',
            'pool accounting is not physical GPU residency; RSS may count shared mappings more than once',
            'delivered text chunks are not necessarily one token each'],
    },
    'lifecycleStages': [row['stage'] for row in rows],
    'successorNumericalComparison': read('successor-numerical-comparison.json'),
    'pending': ['physical Vulkan qualification of this successor', 'optional M3 27B lane',
        'separate scratch-lifetime campaign before assuming stable input/output bindings',
        'no optimized package or general performance claim'],
}
assert summary['sameRecorderIdentityRepeats'] == 0
assert summary['successorNumericalComparison']['passed']
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ['decision', 'recordedDecodeMatmulCalls', 'sameRecorderIdentityRepeats', 'lifetimeIdentityRepeats', 'baselineObservations']}, indent=2))
