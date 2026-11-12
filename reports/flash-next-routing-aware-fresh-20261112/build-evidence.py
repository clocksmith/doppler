"""Assemble doppler.routing-aware-acceptance-evidence/v1 and the task-quality file.

Usage: python3 -P build-evidence.py ROOT PLAN.json POLICY_SHA256 PROMPT_SET.json

ROOT layout (written by the fresh-test procedure in README.md):
  frozen/inputs.json                 frozen prompt tokens
  cpu/<group>/cpu-greedy.json        CPU greedy continuations (cpu-reference.py chain)
  cpu/<group>/round-0/               CPU reference, router rows, <case>-sampled-logits.npy
  fixture/<case>/fixture.json        prepare-routing-fixture.py output
  gpu/observe/<case>/                unforced run with executed-selection readback
  gpu/plain/<group>/observation.json uninstrumented run
  gpu/indices/<case>/                forced-indices control
  gpu/forced-comparison/<case>.json  compare-forced-routing.py on the forced run
  gpu/task-quality/                  flash-next-task-quality-physical.test.js output
  same-input/audit.json              audit-same-input-all-layers.py output
Writes ROOT/evidence/{evidence.json,task-quality.json,build-notes.json}. Missing parts are
left out, so the evaluator reports them; this script never decides a verdict.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

if len(sys.argv) != 5:
    raise SystemExit(__doc__)
root, plan_path = Path(sys.argv[1]), Path(sys.argv[2])
policy_sha256, prompt_set_path = sys.argv[3], Path(sys.argv[4])
plan = json.loads(plan_path.read_bytes())
prompt_set = json.loads(prompt_set_path.read_bytes())
sha = lambda raw: hashlib.sha256(raw).hexdigest()
out = root / 'evidence'
out.mkdir(exist_ok=True)
rel = lambda path: os.path.relpath(path, out)
notes = []


def load(path):
    return json.loads(Path(path).read_bytes()) if Path(path).exists() else None


manifest = json.loads((Path(plan['modelDir']) / 'manifest.json').read_bytes())
inputs = {row['id']: row for row in json.loads((root / 'frozen' / 'inputs.json').read_bytes())['sequences']}
group_of = {case: group for group, cases in plan['groups'].items() for case in cases}
adapters = set()


def adapter_string(info):
    if not info:
        return None
    return ' | '.join(str(info.get(key, '')) for key in ['vendor', 'architecture', 'device', 'description'])


def tuple_for(info):
    pinned = plan['qualificationTuple']
    adapter = adapter_string(info)
    if adapter:
        adapters.add(adapter)
    return {'modelId': manifest['modelId'], 'artifactManifestSha256': plan['expectedManifestSha256'],
            'precision': pinned['precision'], 'backend': pinned['backend'],
            'device': {'vendor': info.get('vendor') if info else None,
                       'architecture': (info.get('architecture') or '').replace('-', '') if info else None,
                       'adapter': adapter}}


starts = []
cases = []
for case in prompt_set['cases']:
    cid = case['id']
    group = group_of[cid]
    tokens = len(inputs[cid]['tokens'])
    item = {'caseId': cid, 'numLayers': 48, 'numTokens': tokens, 'numExperts': 512, 'topK': 10}
    fixture = root / 'fixture' / cid / 'fixture.json'
    if fixture.exists():
        item['cpu'] = {'fixture': rel(fixture)}
    observe_dir = root / 'gpu' / 'observe' / cid
    control = load(observe_dir / 'control.json')
    observation = load(observe_dir / 'observation.json')
    plain = load(root / 'gpu' / 'plain' / group / 'observation.json')
    if control and control.get('completed') and observation:
        row = next(r for r in observation['rows'] if r['id'] == cid)
        plain_row = next((r for r in (plain or {}).get('rows', []) if r['id'] == cid), None)
        layers = sorted(control['topKLayers'], key=lambda r: r['layer'])
        item['unforced'] = {
            'mode': control['mode'], 'selectionSource': 'gpu-executed-readback', 'tuple': tuple_for(control.get('adapterInfo')),
            'prefillLogitsSha256': row['prefillLogitsSha256'],
            'uninstrumentedPrefillLogitsSha256': plain_row['prefillLogitsSha256'] if plain_row else None,
            'layers': [{'layer': r['layer'], 'experts': {'file': rel(observe_dir / 'selections' / r['executedExperts']['file']),
                                                          'sha256': r['executedExperts']['sha256']}} for r in layers]}
        if not plain_row:
            notes.append(f'{cid}: plain run missing')
    else:
        notes.append(f'{cid}: unforced observe run missing or incomplete')
    indices_dir = root / 'gpu' / 'indices' / cid
    forced = load(indices_dir / 'control.json')
    comparison = load(root / 'gpu' / 'forced-comparison' / f'{cid}.json')
    if forced and forced.get('completed'):
        logits = None
        if comparison:
            crow = comparison['rows'][0]
            logits = {'policyId': comparison['policyId'], 'count': crow['count'], 'outsidePolicy': crow['outsidePolicy'],
                      'maxAbsoluteError': crow['maxAbsoluteError']}
        item['forcedIndices'] = {
            'mode': forced['mode'], 'completed': True, 'tuple': tuple_for(forced.get('adapterInfo')),
            'executedSelectionDifferingTokenLayers': sum(r['selectedSetDifferingTokens'] for r in forced['topKLayers']),
            'logits': logits,
            'layers': [{'layer': r['layer'], 'routerLogitsMaxAbsDiffVsReference': r['routerLogitsMaxAbsDiffVsReference'],
                        'ownSelectionDifferences': r['ownSelectionDifferences']}
                       for r in sorted(forced['routerLayers'], key=lambda r: r['layer'])]}
    else:
        notes.append(f'{cid}: forced-indices control missing or incomplete')
    cases.append(item)

tq = load(root / 'gpu' / 'task-quality' / 'observation.json')
if tq:
    starts.append(tq['runStartedAtUtc'])
stop = set(manifest['eos_token_id'] if isinstance(manifest['eos_token_id'], list) else [manifest['eos_token_id']])


def truncate(tokens):
    for i, token in enumerate(tokens):
        if token in stop:
            return tokens[:i + 1]
    return tokens


task_quality = None
if tq:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str((Path(plan['modelDir']) / 'tokenizer.json').read_text())
    rows, answers, counts = [], [], {}
    gpu_cases = {row['caseId']: row for row in tq['cases']}
    for case in prompt_set['cases']:
        cid = case['id']
        counts[cid] = len(inputs[cid]['tokens'])
        group = group_of[cid]
        greedy = load(root / 'cpu' / group / 'cpu-greedy.json')
        cpu_rows_path = root / 'cpu' / group / 'round-0' / f'{cid}-sampled-logits.npy'
        g = gpu_cases.get(cid)
        if g and cpu_rows_path.exists():
            cpu_rows = np.load(cpu_rows_path).astype(np.float64)
            gpu_raw = (root / 'gpu' / 'task-quality' / g['sampledLogits']['file']).read_bytes()
            assert sha(gpu_raw) == g['sampledLogits']['sha256']
            gpu_rows = np.frombuffer(gpu_raw, dtype='<f4').reshape(g['sampledLogits']['shape']).astype(np.float64)
            assert cpu_rows.shape == gpu_rows.shape, (cid, cpu_rows.shape, gpu_rows.shape)
            for position, c, q in zip(g['positions'], cpu_rows, gpu_rows):
                lc = c - c.max(); lc -= np.log(np.exp(lc).sum())
                lq = q - q.max(); lq -= np.log(np.exp(lq).sum())
                top2 = np.partition(c, -2)[-2:]
                rows.append({'caseId': cid, 'position': int(position), 'cpuTop1': int(c.argmax()), 'gpuTop1': int(q.argmax()),
                             'cpuTop1MarginLogits': float(top2.max() - top2.min()),
                             'klCpuGpuNats': float(np.sum(np.exp(lc) * (lc - lq)))})
        else:
            notes.append(f'{cid}: sampled-position logits missing')
        cpu_case = (greedy or {}).get('cases', {}).get(cid)
        complete = cpu_case and (cpu_case['stopped'] or len(cpu_case['tokens']) == prompt_set['generation']['maxNewTokens'])
        if g and complete:
            cpu_tokens, gpu_tokens = truncate(cpu_case['tokens']), truncate(g['gpuTokens'])
            answers.append({'caseId': cid, 'cpuTokens': cpu_tokens, 'gpuTokens': gpu_tokens,
                            'cpuText': tokenizer.decode(cpu_tokens, skip_special_tokens=True),
                            'gpuText': tokenizer.decode(gpu_tokens, skip_special_tokens=True),
                            'gpuHarnessText': g['gpuText']})
        else:
            notes.append(f'{cid}: greedy answer pair incomplete')
    task_quality = {'schema': 'doppler.routing-aware-task-quality/v1', 'policySha256': policy_sha256,
                    'promptSetSha256': plan['promptSetSha256'], 'runStartedAtUtc': tq['runStartedAtUtc'],
                    'tuple': tuple_for(tq.get('adapterInfo')), 'caseTokenCounts': counts,
                    'distributionRows': rows, 'answers': answers}
    raw = (json.dumps(task_quality, indent=2) + '\n').encode()
    (out / 'task-quality.json').write_bytes(raw)

same = load(root / 'same-input' / 'audit.json')
capture = load(root / 'same-input' / 'capture' / 'observation.json')
if capture:
    starts.append(capture['runStartedAtUtc'])
# ROOT/run-starts.jsonl: one {"run", "startedAtUtc"} line written before each GPU or CPU job.
if (root / 'run-starts.jsonl').exists():
    starts += [json.loads(line)['startedAtUtc'] for line in (root / 'run-starts.jsonl').read_text().splitlines() if line]
evidence = {'schema': 'doppler.routing-aware-acceptance-evidence/v1', 'policySha256': policy_sha256,
            'promptSetSha256': plan['promptSetSha256'], 'runStartedAtUtc': min(starts) if starts else None,
            'tuple': tuple_for(tq.get('adapterInfo') if tq else None), 'cases': cases,
            'sameInputOperators': ({'tuple': tuple_for(capture.get('adapterInfo')), 'policyId': same['policyId'],
                                    'checks': [{k: c[k] for k in ['family', 'caseId', 'layers', 'count', 'outsidePolicy']}
                                               for c in same['checks']]} if same and capture else None),
            'taskQuality': ({'file': 'task-quality.json', 'sha256': sha((out / 'task-quality.json').read_bytes())}
                            if task_quality else None)}
(out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
(out / 'build-notes.json').write_text(json.dumps({'adapterStrings': sorted(adapters), 'notes': notes,
                                                  'builderSha256': sha(Path(__file__).read_bytes())}, indent=2) + '\n')
print(json.dumps({'cases': len(cases), 'adapters': sorted(adapters), 'notes': len(notes)}))
