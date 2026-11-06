"""Verify final runtime bytes against the retained bank controls."""
import json
from pathlib import Path
import sys
from compare import compare, load


def main(directory):
    root = Path(directory)
    old, old_hash = load(root / 'moe-layer1-stages-r2-20261106.json')
    new, new_hash = load(root / 'routed-moe-layer1-r5-20261106.json')
    for key in ['manifestSha256', 'inputsSha256']:
        if old[key] != new[key]:
            raise ValueError(f'Unmatched trace identity: {key}')
    pairs = [('moeInput', old['moeInput'], new['moeInput']),
             ('moeOutput', old['moeOutput'], new['moeOutput'])]
    pairs += [(key, old['moeRouting'][key], new['moeRouting'][key])
              for key in ['indices', 'weights']]
    rows = [dict(boundary=label, **compare(a, b)) for label, a, b in pairs]
    baseline, baseline_hash = load(root / 'routed-cost-bank-20261106.json')
    final, final_hash = load(root / 'routed-cost-final-20261106.json')
    if baseline['manifestSha256'] != final['manifestSha256']:
        raise ValueError('Unmatched generation artifacts')
    if baseline['runtimeConfig']['inference'] != final['runtimeConfig']['inference']:
        raise ValueError('Inference policy changed')
    if len(baseline['rows']) != len(final['rows']):
        raise ValueError('Request counts differ')
    for i, (a, b) in enumerate(zip(baseline['rows'], final['rows'])):
        if a['tokenIds'] != b['tokenIds'] or len(a['logits']) != len(b['logits']):
            raise ValueError('Final generation differs')
        for j, (left, right) in enumerate(zip(a['logits'], b['logits'])):
            rows.append(dict(repetition=i, phase=j, **compare(
                {**left, 'shape': [left['count']]},
                {**right, 'shape': [right['count']]})))
    check, check_hash = load(root / 'routed-cost-bank-counter-check-20261106.json')
    count = check['rows'][0]['after']
    if count['expertTensorLoads'] != 96 or count['expertTensorBytes'] != 72477573120:
        raise ValueError('Observer did not bind all 48 expert banks')
    for report in [new, final, check]:
        if (report.get('gpuErrors') or report.get('cleanupFailures')
                or not (report.get('completed') or report.get('diagnosticCompleted'))):
            raise ValueError('Incomplete or failed physical observation')
    if final['providerBefore'] != final['providerAfter']:
        raise ValueError('Existing provider changed')
    result = dict(scope='Final runtime parity, lifecycle and corrected observer; timing pair retained separately',
                  passed=True, rows=rows,
                  reportSha256=dict(oldTrace=old_hash, finalTrace=new_hash,
                                    bank=baseline_hash, final=final_hash, counterCheck=check_hash),
                  lifecycle=[s['case'] for s in final['lifecycle']],
                  counterCheck=dict(loads=count['expertTensorLoads'], expertPayloadBytes=count['expertTensorBytes']),
                  providerBefore=final['providerBefore'], providerAfter=final['providerAfter'],
                  runtimeSourceSha256=new['runtimeSourceSha256'])
    (root / 'routed-final-verification-20261106.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, bitExactRows=len(rows), lifecycle=result['lifecycle'])))


if __name__ == '__main__':
    main(sys.argv[1])
