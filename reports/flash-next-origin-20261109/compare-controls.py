"""Reproduce numeric diagnostics, never accept substituted model execution.

Usage: python3 compare-controls.py PROTOCOL REFERENCE CPU_ROUTING_TRACE POLICY
       OPERATOR_DATA_ROOT NEW_OUTPUT
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

protocol_path, reference_path, cpu_root, policy_path, root, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
protocol_bytes, reference_bytes, cpu_bytes, policy_bytes = [p.read_bytes() for p in [
    protocol_path, reference_path, cpu_root / 'receipt.json', policy_path]]
protocol, reference, cpu, policies = map(json.loads, [protocol_bytes, reference_bytes, cpu_bytes, policy_bytes])
bindings = protocol['bindings']
assert sha(reference_bytes) == bindings['referenceSha256']
assert sha(cpu_bytes) == bindings['cpuRoutingTraceReceiptSha256']
assert sha(policy_bytes) == bindings['policySha256']
assert cpu['completed'] and cpu['samplesAndRouterMetadataMatchReference']
assert reference['inputsSha256'] == cpu['inputsSha256'] == bindings['inputsSha256']
assert reference['artifactManifestSha256'] == cpu['manifestSha256'] == bindings['manifestSha256']
policy_id = 'doppler.boundary-tolerance/quantized-v1'
policy = policies['tolerancePolicies'][policy_id]
expected = reference['sequences']['long']
file = Path(expected['logitsFile'])
if not file.is_absolute():
    file = reference_path.parent / file
assert sha(file.read_bytes()) == expected['logitsSha256']
full = np.load(file, allow_pickle=False)
source = full[expected['samplePositions'].index(2303)].astype('f8')
assert source.shape == (248320,) and np.isfinite(source).all()
results, source_routers = [], {}
baseline = json.loads((root / 'counterfactual-baseline/observation.json').read_bytes())

for mode in ['baseline', 'first-down', 'first-hc', 'prefix-state']:
    directory = root / ('counterfactual-' + mode)
    observed_bytes = (directory / 'observation.json').read_bytes()
    observed = json.loads(observed_bytes)
    assert observed['completed'] and not observed['gpuErrors'] and not observed['cleanupFailures']
    assert not observed.get('routingControl')
    assert observed['manifestSha256'] == bindings['manifestSha256']
    assert observed['inputsSha256'] == bindings['inputsSha256']
    assert observed['referenceSha256'] == bindings['referenceSha256']
    assert observed['runtimeConfig'] == baseline['runtimeConfig']
    control = observed['counterfactual']
    assert control['completed'] and control['mode'] == mode and control['normalModelQualified'] is False
    replaced = control['replaced']
    expected_replacements = {'baseline': [], 'first-down': [(0, 'down')],
                             'first-hc': [(0, 'branchInput'), (0, 'gate')], 'prefix-state': [(24, 'input')]}[mode]
    assert [(r['layer'], r['stage']) for r in replaced] == expected_replacements
    assert all(r['writeVerified'] for r in replaced)
    assert len(observed['rows']) == 1
    row = observed['rows'][0]
    assert row['id'] == 'long' and row['inputTokenCount'] == 2304 and row['kvStorageDtype'] == 'f32'
    assert row['logitRows'] == 1
    raw = Path(row['prefillLogitsFile']).read_bytes()
    assert sha(raw) == row['prefillLogitsSha256'] and len(raw) == source.size * 4
    actual = np.frombuffer(raw, '<f4').astype('f8')
    assert np.isfinite(actual).all()
    delta = np.abs(actual - source)
    outside = int(np.count_nonzero(delta > policy['absolute'] + policy['relative'] * np.abs(source)))
    routing = None
    if mode != 'baseline':
        routes_bytes = (directory / 'actual-routes.json').read_bytes()
        routes = json.loads(routes_bytes)
        rows = []
        for record in routes['records']:
            layer = record['layer']
            raw = Path(record['indices']['file']).read_bytes()
            assert sha(raw) == record['indices']['sha256']
            indices = np.frombuffer(raw, '<u4').reshape(2304, 10)
            assert (indices < 512).all()
            if layer not in source_routers:
                name = f'router-{layer}.npz'
                assert sha((cpu_root / name).read_bytes()) == cpu['files'][name]
                with np.load(cpu_root / name, allow_pickle=False) as data:
                    source_routers[layer] = data['experts'].copy()
            positions = np.flatnonzero(np.any(np.sort(indices, axis=-1)
                                             != np.sort(source_routers[layer], axis=-1), axis=-1)).tolist()
            order_differences = int(np.count_nonzero(np.any(indices != source_routers[layer], axis=-1)))
            rows.append(dict(layer=layer, expertSetDifferences=len(positions), positions=positions,
                             expertOrderDifferences=order_differences))
        routing = dict(actualRoutesSha256=sha(routes_bytes), inspectedLayers=[r['layer'] for r in rows],
                       tokenLayerDecisions=2304 * len(rows), differences=[r for r in rows if r['expertSetDifferences']],
                       expertOrderDifferences=sum(r['expertOrderDifferences'] for r in rows))
    results.append(dict(mode=mode, counterfactual=bool(replaced), normalModelQualified=False,
                        observationSha256=sha(observed_bytes), prefillLogitsSha256=row['prefillLogitsSha256'],
                        observedLogits=int(actual.size), maxAbsoluteError=float(delta.max()),
                        rmsError=float(np.sqrt(np.mean(delta * delta))), outsideUnchangedPolicy=outside,
                        firstTokenAgrees=int(np.argmax(actual)) == int(expected['argmax'][2303]) == row['tokenId'],
                        routing=routing, replacements=[{k: r[k] for k in [
                            'layer', 'role', 'stage', 'shape', 'originalSha256', 'fixtureSha256', 'writeVerified']}
                            for r in replaced]))

assert results[0]['outsideUnchangedPolicy'] == 13, 'Frozen failure was not reproduced'
report = dict(scope='Full-network substituted-execution diagnostics; never normal-model qualification',
              normalModelQualified=False, qualificationAuthority=False,
              originalFailureRemainsFailed=True, policyId=policy_id, policySha256=sha(policy_bytes),
              manifestSha256=bindings['manifestSha256'], inputsSha256=bindings['inputsSha256'],
              referenceSha256=sha(reference_bytes), cpuRoutingTraceReceiptSha256=sha(cpu_bytes),
              protocolSha256=sha(protocol_bytes), comparatorSha256=sha(Path(__file__).read_bytes()), rows=results)
assert not output.exists()
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps([dict(mode=r['mode'], maxAbsoluteError=r['maxAbsoluteError'],
                       outsideUnchangedPolicy=r['outsideUnchangedPolicy']) for r in results], indent=2))
