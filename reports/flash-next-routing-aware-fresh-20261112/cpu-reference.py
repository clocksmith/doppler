"""CPU F32 reference for the routing-aware fresh test: pinned producer, added observations only.

Usage:
  python3 -P cpu-reference.py round PRODUCER SOURCE ARTIFACT PROMPT_SET INPUTS OUT_DIR SEQ[,SEQ...] [--observe]
  python3 -P cpu-reference.py chain PRODUCER SOURCE ARTIFACT PROMPT_SET INPUTS OUT_DIR SEQ[,SEQ...]

`round` executes the pinned producer (artifacts/qwen38-flash-next-20261013/
prepare-learned-forward-reference.py, SHA-256 checked) on INPUTS for the named
sequences, with --artifact so it reads stored F16/Q4_K weights. With --observe it
adds two observations and no arithmetic: every layer's router logits, experts and
scores for every position (router-<seq>/router-<L>.npz plus a receipt in the format
prepare-routing-fixture.py reads), and the complete logit rows at the prompt set's
sampled positions (<seq>-sampled-logits.npy).

`chain` runs round 0 with --observe on the frozen prompt tokens, then greedy decoding:
each later round re-runs the unmodified producer, cache-free, on prompt + the CPU's own
earlier greedy tokens, until maxNewTokens tokens or a stop token. It never reads GPU
output. Writes OUT_DIR/cpu-greedy.json.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PINNED_PRODUCER_SHA256 = 'f6bc0f15a6545ab1d205cfc3ec412442415220e541190b63a9c712fa238ba5b0'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sample_positions(rule, tokens):
    positions = list(range(rule['offset'], tokens, rule['stride']))
    if rule['includeLast'] and (not positions or positions[-1] != tokens - 1):
        positions.append(tokens - 1)
    return positions


def instrument(text):
    def once(anchor, replacement):
        nonlocal text
        require(text.count(anchor) == 1, f'Ambiguous producer boundary: {anchor}')
        text = text.replace(anchor, replacement)

    anchor = "    rec['minMargin'] = float(min(rec['marginKthToNext']))"
    once(anchor, anchor + '''
    _router_dir = OUT / f"router-{current['seq']}"
    _router_dir.mkdir(exist_ok=True)
    np.savez(_router_dir / f"router-{current['layer']}.npz", logits=logits.detach().numpy(),
             experts=output[2].detach().numpy(), scores=output[1].detach().float().numpy())''')
    anchor = "        full_rows = {p: logits[p].numpy() for p in positions}"
    once(anchor, anchor + '''
        np.save(OUT / f"{s['id']}-sampled-logits.npy",
                np.ascontiguousarray(logits[POLICY_POSITIONS[s['id']]].numpy()), allow_pickle=False)''')
    return text


def run_round(producer, source, artifact, prompt_set_path, inputs, out, sequences, observe):
    original = producer.read_bytes()
    require(sha(original) == PINNED_PRODUCER_SHA256, 'Producer differs from the pinned CPU reference producer')
    require(not out.exists(), 'Choose a fresh output directory')
    text = original.decode()
    prompt_set = json.loads(prompt_set_path.read_bytes())
    rows = {row['id']: row for row in json.loads(inputs.read_bytes())['sequences']}
    environment = {'__name__': '__main__', '__file__': str(producer)}
    if observe:
        text = instrument(text)
        environment['POLICY_POSITIONS'] = {seq: sample_positions(prompt_set['samplePositions'], len(rows[seq]['tokens']))
                                           for seq in sequences}
    sys.argv = [str(producer), str(source), str(out), '--dtype', 'float32', '--artifact', str(artifact),
                '--inputs', str(inputs), '--sequences', ','.join(sequences)]
    started = time.time()
    exec(compile(text, str(producer) + (':routing-aware-observation' if observe else ''), 'exec'), environment)
    reference_path = out / 'learned-forward-reference-float32-artifact.json'
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    require(reference['producerSha256'] == PINNED_PRODUCER_SHA256, 'Reference names a different producer')
    if observe:
        import numpy as np
        manifest_sha = sha((artifact / 'manifest.json').read_bytes())
        for seq in sequences:
            directory = out / f'router-{seq}'
            expected = reference['sequences'][seq]
            # Self-consistency: the dumped rows reproduce the reference's own router records.
            for layer in range(48):
                data = np.load(directory / f'router-{layer}.npz')
                record = expected['router'][str(layer)]
                require(data['experts'][record['positions']].tolist() == record['topExperts'], f'{seq} L{layer} experts')
                require(data['logits'].shape == (expected['tokens'], 512), f'{seq} L{layer} shape')
            receipt = dict(scope='Unchanged upstream F32 routing observation recorded in the same execution as the reference; '
                                 'not model acceptance', completed=True, kvStorageDtype='f32', sequenceId=seq,
                           producerSha256=PINNED_PRODUCER_SHA256, instrumentedProducerSha256=sha(text.encode()),
                           instrumentationSha256=sha(Path(__file__).read_bytes()), referenceSha256=sha(reference_bytes),
                           manifestSha256=manifest_sha, inputsSha256=sha(inputs.read_bytes()),
                           samplesAndRouterMetadataMatchReference=True,
                           files={p.name: sha(p.read_bytes()) for p in directory.iterdir() if p.suffix == '.npz'})
            (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            sampled = np.load(out / f'{seq}-sampled-logits.npy')
            require(sampled.shape == (len(environment['POLICY_POSITIONS'][seq]), 248320), f'{seq} sampled rows')
            last = np.load(out / expected['logitsFile'])[expected['samplePositions'].index(expected['tokens'] - 1)]
            require(np.array_equal(sampled[-1].view('u4'), last.view('u4')), f'{seq} sampled last row differs from reference')
    summary = dict(round=dict(observe=observe, sequences=sequences, seconds=round(time.time() - started, 1),
                              referenceSha256=sha(reference_bytes), inputsSha256=sha(inputs.read_bytes())))
    (out / 'round.json').write_text(json.dumps(summary, indent=2) + '\n')
    return reference


def run_chain(script, producer, source, artifact, prompt_set_path, inputs, out, sequences):
    prompt_set = json.loads(prompt_set_path.read_bytes())
    manifest = json.loads((artifact / 'manifest.json').read_bytes())
    stop = set(manifest['eos_token_id'] if isinstance(manifest['eos_token_id'], list) else [manifest['eos_token_id']])
    max_new = prompt_set['generation']['maxNewTokens']
    require(prompt_set['generation']['temperature'] == 0 and prompt_set['generation']['topK'] == 1, 'greedy only')
    out.mkdir()
    prompts = {row['id']: row for row in json.loads(inputs.read_bytes())['sequences']}
    state = {seq: dict(tokens=[], stopped=False, rounds=[]) for seq in sequences}
    for step in range(max_new):
        active = [seq for seq in sequences if not state[seq]['stopped']]
        if not active:
            break
        round_dir = out / f'round-{step}'
        if step == 0:
            round_inputs = inputs
        else:
            round_inputs = out / f'inputs-round-{step}.json'
            rows = [dict(id=seq, text=prompts[seq].get('text'), tokens=prompts[seq]['tokens'] + state[seq]['tokens'])
                    for seq in active]
            round_inputs.write_text(json.dumps(dict(schema='doppler.qwen38-learned-forward-inputs/v1', sequences=rows), indent=2) + '\n')
        command = [sys.executable, '-P', str(script), 'round', str(producer), str(source), str(artifact), str(prompt_set_path),
                   str(round_inputs), str(round_dir), ','.join(active)] + (['--observe'] if step == 0 else [])
        print(json.dumps(dict(step=step, active=active, started=time.strftime('%Y-%m-%dT%H:%M:%S'))), flush=True)
        with open(out / f'round-{step}.log', 'w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        reference_bytes = (round_dir / 'learned-forward-reference-float32-artifact.json').read_bytes()
        reference = json.loads(reference_bytes)
        for seq in active:
            row = reference['sequences'][seq]
            token = int(row['argmax'][row['tokens'] - 1])
            state[seq]['tokens'].append(token)
            state[seq]['rounds'].append(dict(step=step, inputTokens=row['tokens'], token=token,
                                             top1MarginLogits=float(row['top1Margin'][row['tokens'] - 1])))
            if token in stop:
                state[seq]['stopped'] = True
        result = dict(schema='doppler.routing-aware-cpu-greedy/v1', producerSha256=PINNED_PRODUCER_SHA256,
                      wrapperSha256=sha(Path(script).read_bytes()), inputsSha256=sha(inputs.read_bytes()),
                      promptSetFileSha256=sha(prompt_set_path.read_bytes()), stopTokenIds=sorted(stop), maxNewTokens=max_new,
                      completedSteps=step + 1, cases=state)
        (out / 'cpu-greedy.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(completed=True, cases={seq: state[seq]['tokens'] for seq in sequences})), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 9 or sys.argv[1] not in ('round', 'chain'):
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    producer, source, artifact, prompt_set_path, inputs, out = map(Path, sys.argv[2:8])
    sequences = sys.argv[8].split(',')
    require(sequences and len(set(sequences)) == len(sequences), 'Name each sequence once')
    if mode == 'round':
        run_round(producer, source, artifact, prompt_set_path, inputs, out, sequences, '--observe' in sys.argv[9:])
    else:
        run_chain(Path(__file__).resolve(), producer, source, artifact, prompt_set_path, inputs, out, sequences)
