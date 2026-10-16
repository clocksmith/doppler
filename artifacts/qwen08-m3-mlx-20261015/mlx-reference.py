# Run with the isolated environment: mlx-reference.py MODEL_DIR INPUT_JSON OUTPUT_JSON [POLICY_JSON]
import hashlib, importlib.metadata, json, pathlib, resource, sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx.utils import tree_flatten
model_dir, input_path, output_path = map(pathlib.Path, sys.argv[1:4])
policy_path = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else pathlib.Path(__file__).with_name('policy.json')
policy = json.loads(policy_path.read_text())
shared = policy['shared']
assert importlib.metadata.version('mlx-lm') == policy['engines']['mlx']['version']
with (model_dir / policy['source']['weightsFile']).open('rb') as source:
    assert hashlib.file_digest(source, 'sha256').hexdigest() == policy['source']['weightsSha256']
mx.set_memory_limit(shared['memoryLimitBytes'])
record = {'engine': 'mlx', 'surface': 'native-metal', 'policy': policy, 'versions': {n: importlib.metadata.version(n) for n in ['mlx', 'mlx-lm']}, 'runs': [], 'complete': False}
try:
    start = time.perf_counter()
    model, tokenizer = load(str(model_dir))
    tokenizer.eos_token_ids = shared['eosTokenIds']
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    record['modelLoadMs'] = (time.perf_counter() - start) * 1000
    record['device'] = mx.device_info()
    record['parameterBytesByDtype'] = {}
    for name, tensor in tree_flatten(model.parameters()):
        dtype = str(tensor.dtype)
        record['parameterBytesByDtype'][dtype] = record['parameterBytesByDtype'].get(dtype, 0) + tensor.nbytes
    if input_path.exists():
        inputs = json.loads(input_path.read_text())
    else:
        text = shared['promptText'] * 100
        ids = tokenizer.encode(text)
        assert len(ids) >= max(shared['promptTokenCounts'])
        inputs = {'source': policy['source'], 'prompts': [{'tokenIds': ids[:n]} for n in shared['promptTokenCounts']]}
        input_path.write_text(json.dumps(inputs, indent=2) + '\n')
    assert inputs['source'] == policy['source'], 'Prompt source mismatch'
    assert [len(p['tokenIds']) for p in inputs['prompts']] == shared['promptTokenCounts'], 'Prompt lengths mismatch'
    def suppress(_tokens, logits):
        logits[..., shared['suppressTokenIds']] = -float('inf')
        return logits
    for prompt in inputs['prompts']:
        for i in range(shared['warmupRuns'] + shared['timedRuns']):
            mx.reset_peak_memory()
            started = time.perf_counter()
            times, tokens, text = [], [], ''
            for response in stream_generate(model, tokenizer, prompt=prompt['tokenIds'], max_tokens=shared['maxTokens'], sampler=make_sampler(temp=shared['temperature']), logits_processors=[suppress] if shared['suppressTokenIds'] else [], prefill_step_size=policy['engines']['mlx']['prefillStepSize']):
                times.append((time.perf_counter() - started) * 1000)
                tokens.append(response.token)
                text += response.text
            mx.synchronize()
            record['runs'].append({'promptTokenIds': prompt['tokenIds'], 'warmup': i < shared['warmupRuns'], 'totalRunMs': (time.perf_counter() - started) * 1000, 'firstTokenMs': times[0] if times else None, 'tokenTimes': times, 'tokenIds': tokens, 'outputText': text, 'memory': {'mlxPeakBytes': mx.get_peak_memory(), 'mlxActiveBytes': mx.get_active_memory(), 'processPeakRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}})
            output_path.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({'promptTokens': len(prompt['tokenIds']), 'run': i, 'tokens': len(tokens), 'totalRunMs': record['runs'][-1]['totalRunMs']}), flush=True)
    record['complete'] = True
except Exception as error:
    record['error'] = str(error)
    raise
finally:
    output_path.write_text(json.dumps(record, indent=2) + '\n')
