"""Resume the frozen cpu-reference.py greedy chain from its last completed step.

Usage: python3 -P resume-cpu-chain.py SCRIPT PRODUCER SOURCE ARTIFACT PROMPT_SET INPUTS OUT

Mirrors run_chain() in cpu-reference.py (b826337d): same per-round command line (the frozen script's
`round` mode), same input construction, same stop rule and the same cpu-greedy.json fields. It reads the
completed steps from OUT/cpu-greedy.json; a partial round directory for the next step must have been moved
aside first. Never reads GPU output.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

if len(sys.argv) != 8:
    raise SystemExit(__doc__)
script, producer, source, artifact, prompt_set_path, inputs, out = map(Path, sys.argv[1:8])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
prompt_set = json.loads(prompt_set_path.read_bytes())
manifest = json.loads((artifact / 'manifest.json').read_bytes())
stop = set(manifest['eos_token_id'] if isinstance(manifest['eos_token_id'], list) else [manifest['eos_token_id']])
max_new = prompt_set['generation']['maxNewTokens']
prior = json.loads((out / 'cpu-greedy.json').read_bytes())
assert prior['inputsSha256'] == sha(inputs.read_bytes()) and prior['wrapperSha256'] == sha(script.read_bytes())
state = prior['cases']
sequences = list(state)
prompts = {row['id']: row for row in json.loads(inputs.read_bytes())['sequences']}
for step in range(prior['completedSteps'], max_new):
    active = [seq for seq in sequences if not state[seq]['stopped']]
    if not active:
        break
    round_dir = out / f'round-{step}'
    assert not round_dir.exists(), f'{round_dir} exists; move the partial round aside'
    round_inputs = out / f'inputs-round-{step}.json'
    rows = [dict(id=seq, text=prompts[seq].get('text'), tokens=prompts[seq]['tokens'] + state[seq]['tokens'])
            for seq in active]
    round_inputs.write_text(json.dumps(dict(schema='doppler.qwen38-learned-forward-inputs/v1', sequences=rows), indent=2) + '\n')
    command = [sys.executable, '-P', str(script), 'round', str(producer), str(source), str(artifact), str(prompt_set_path),
               str(round_inputs), str(round_dir), ','.join(active)]
    print(json.dumps(dict(step=step, active=active, started=time.strftime('%Y-%m-%dT%H:%M:%S'), resumed=True)), flush=True)
    with open(out / f'round-{step}.log', 'w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    reference = json.loads((round_dir / 'learned-forward-reference-float32-artifact.json').read_bytes())
    for seq in active:
        row = reference['sequences'][seq]
        token = int(row['argmax'][row['tokens'] - 1])
        state[seq]['tokens'].append(token)
        state[seq]['rounds'].append(dict(step=step, inputTokens=row['tokens'], token=token,
                                         top1MarginLogits=float(row['top1Margin'][row['tokens'] - 1])))
        if token in stop:
            state[seq]['stopped'] = True
    result = dict(schema='doppler.routing-aware-cpu-greedy/v1', producerSha256=prior['producerSha256'],
                  wrapperSha256=sha(script.read_bytes()), inputsSha256=sha(inputs.read_bytes()),
                  promptSetFileSha256=sha(prompt_set_path.read_bytes()), stopTokenIds=sorted(stop), maxNewTokens=max_new,
                  completedSteps=step + 1, cases=state)
    (out / 'cpu-greedy.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(completed=True, cases={seq: state[seq]['tokens'] for seq in sequences})), flush=True)
