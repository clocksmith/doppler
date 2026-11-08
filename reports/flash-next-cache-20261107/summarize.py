"""Verify every retained full-logit file and summarize observed cache costs.

Usage: python3 summarize.py EXTERNAL_RECEIPT_ROOT OUTPUT_JSON
The root has gpu-budget, generic-host, expert-all and expert-repeat directories.
Raw tensor observations stay on the qualification volume, outside Git.
"""
from pathlib import Path
import hashlib
import json
import sys

root, output = Path(sys.argv[1]), Path(sys.argv[2])
groups = {
    'gpu-budget': ['a1', 'b1', 'b2', 'a2', 'c1', 'd1', 'd2', 'c2'],
    'generic-host': ['e1', 'f1', 'f2', 'e2'],
    'expert-all': ['g1', 'h1', 'h2', 'g2'],
    'expert-repeat': ['i1', 'j1', 'j2', 'i2'],
}
report = dict(schema='doppler.flash-next-cache-comparison/v1',
              scope='Single shared AMD host, deterministic raw prompt, full-logit observation. OS caches uncontrolled.',
              fullModelQualified=False, rows=[], verifiedLogitRows=0, verifiedValues=0)
expected = None
for group, samples in groups.items():
    for sample in samples:
        path = root / group / (sample + '.json')
        raw = path.read_bytes()
        run = json.loads(raw)
        assert run['completed'] and not run['gpuErrors'] and not run['cleanupFailures'], str(path)
        assert run['providerAfter'] == run['providerBefore']
        for row in run['rows']:
            phases, before, after = row['phases'], row['before'], row['after']
            assert len(phases) == len(row['logits']) == len(row['tokenIds']) == 16
            observed = []
            for item in row['logits']:
                binary = path.parent / Path(item['file']).name
                payload = binary.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                assert digest == item['sha256'] and len(payload) == item['count'] * 4
                assert item['count'] == 248320
                observed.append(digest)
                report['verifiedLogitRows'] += 1
                report['verifiedValues'] += item['count']
            identity = dict(manifest=run['manifestSha256'], tokens=row['tokenIds'], logits=observed)
            if expected is None:
                expected = identity
            assert identity == expected, f'Full-logit parity failed: {path}, repeat {row["repetition"]}'
            b = before['routed'] or {}
            a = after['routed']
            diff = lambda key: a[key] - b.get(key, 0)
            intervals = len(phases) - 1
            decode_ms = phases[-1]['atMs'] - phases[0]['atMs']
            report['rows'].append(dict(
                group=group, sample=sample, repetition=row['repetition'],
                receiptSha256=hashlib.sha256(raw).hexdigest(),
                requestMs=after['atMs'] - before['atMs'],
                prefillToObservedLogitsMs=phases[0]['atMs'] - before['atMs'],
                decodeIntervals=intervals, decodeMs=decode_ms,
                msPerDecodeInterval=decode_ms / intervals, tokensPerSecond=1000 * intervals / decode_ms,
                sourceRangeBytes=diff('sourceReadBytes'), uploadBytes=diff('uploadBytes'),
                physicalReadBytes=after['disk']['read_bytes'] - before['disk']['read_bytes'],
                hostBytes=a.get('hostCacheBytes', 0), hostHits=diff('hostCacheHits') if 'hostCacheHits' in a else None,
                peakHostOwnedOrReservedBytes=a.get('peakHostOwnedOrReservedBytes', 0),
                peakAccountedGpuBytes=a['peakAccountedGpuBytes'], rssBytes=after['memory']['rss'],
                tokenIds=row['tokenIds'], text=row['text']))
report['parity'] = 'All actual raw files rehashed; token IDs and all full-vocabulary logit bytes identical.'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(requests=len(report['rows']), rows=report['verifiedLogitRows'], values=report['verifiedValues'])))
