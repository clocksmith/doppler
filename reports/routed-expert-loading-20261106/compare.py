"""Verify observed bytes before summarizing the two explicit loading modes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flash-next-qualification-20261106'))
from verified_tensor_io import read_tensor


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def compare(left, right):
    a, b = read_tensor(left), read_tensor(right)
    if a.shape != b.shape:
        raise ValueError('Unmatched tensor shapes')
    exact = np.array_equal(a.view('u4'), b.view('u4'))
    row = dict(elements=a.size, bitExact=bool(exact),
               maxAbsError=float(np.max(np.abs(a.astype('f8') - b.astype('f8')))),
               baselineSha256=left['sha256'], routedSha256=right['sha256'])
    if not exact:
        raise ValueError(f'Loading changed execution: {row}')
    return row


def summarize(row):
    before, after = row['before'], row['after']
    times = [before['atMs']] + [p['atMs'] for p in row['phases']]
    phases = [dict(index=i, ms=end-start,
                   uploadBytes=point['uploadBytes']-previous['uploadBytes'],
                   storageApiBytes=point['source']['bytesRead']-previous['source']['bytesRead'])
              for i, (start, end, previous, point) in enumerate(zip(
                  times, times[1:], [before]+row['phases'][:-1], row['phases']))]
    delta = lambda key: after[key]-before[key]
    routed = None
    if after['routed']:
        routed = {key: value-(before['routed'] or {}).get(key, 0)
                  for key, value in after['routed'].items()
                  if key in ['sourceReadBytes', 'uploadBytes', 'sourceReads', 'cacheHits',
                             'cacheMisses', 'workingHits', 'evictions', 'ioWaitMs',
                             'sourceReadWaitMs', 'uploadEnqueueMs', 'initialQueueIdleForWeightsMs']}
    return dict(cache=row['cache'], elapsedMs=delta('atMs'), phases=phases,
                uploadBytes=delta('uploadBytes'),
                storageApiBytes=after['source']['bytesRead']-before['source']['bytesRead'],
                physicalDiskReadBytes=after['disk']['read_bytes']-before['disk']['read_bytes'],
                processReadChars=after['disk']['rchar']-before['disk']['rchar'],
                routed=routed, routedAfter=after['routed'], pool=after['pool'],
                maxRssKiB=after['maxRssKiB'], tokenIds=row['tokenIds'], text=row['text'])


def main(bank_path, routed_path, output):
    bank, bank_hash = load(bank_path)
    routed, routed_hash = load(routed_path)
    if bank['mode'] != 'bank' or routed['mode'] != 'routed':
        raise ValueError('Expected bank then routed observations')
    for key in ['manifestSha256', 'prompt', 'sourceSha256']:
        if bank[key] != routed[key]:
            raise ValueError(f'Unmatched comparison identity: {key}')
    if bank['runtimeConfig']['inference'] != routed['runtimeConfig']['inference']:
        raise ValueError('Inference policies differ')
    for report in [bank, routed]:
        if not report.get('completed') or report['gpuErrors'] or report['cleanupFailures']:
            raise ValueError('Incomplete or failed observation')
        if report['providerBefore'] != report['providerAfter']:
            raise ValueError('Provider changed')
    if len(bank['rows']) != len(routed['rows']):
        raise ValueError('Request counts differ')
    parity = []
    for index, (left, right) in enumerate(zip(bank['rows'], routed['rows'])):
        if left['tokenIds'] != right['tokenIds'] or len(left['logits']) != len(right['logits']):
            raise ValueError('Generation differs')
        for phase, (a, b) in enumerate(zip(left['logits'], right['logits'])):
            parity.append(dict(repetition=index, phase=phase, **compare(
                {**a, 'shape': [a['count']]}, {**b, 'shape': [b['count']]})))
    report = dict(scope='One shared-host diagnostic pair with full-logit observation; not a throughput promotion or full-model qualification',
                  bankReportSha256=bank_hash, routedReportSha256=routed_hash,
                  parity=parity, bank=[summarize(r) for r in bank['rows']],
                  routed=[summarize(r) for r in routed['rows']],
                  lifecycle=routed.get('lifecycle', []), passed=True)
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(parityRows=len(parity), bitExact=True,
                         bankMs=[r['elapsedMs'] for r in report['bank']],
                         routedMs=[r['elapsedMs'] for r in report['routed']])))


if __name__ == '__main__':
    main(*sys.argv[1:])
