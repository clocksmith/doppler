"""One exact real-model projection under encryption; all other model execution remains client-owned."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
import concrete_ml_extensions as ext
from bridge import EvaluatorBridge
from linear import ClientLinearProjection
from profiles import load_profile
from protocol import POLICY, digest


def validate_source(directory, specification):
    for name, identity in specification['source']['files'].items():
        try:
            with (directory / name).open('rb') as source:
                actual = hashlib.file_digest(source, 'sha256').hexdigest()
        except OSError:
            raise ValueError('Pinned real-model artifact required') from None
        if actual != identity:
            raise ValueError('Pinned real-model artifact required')


def fixed_inputs(tokens, specification):
    if not 1 <= len(tokens) <= specification['contextLength']:
        raise ValueError('Unsupported private context bucket')
    count = specification['contextLength'] - len(tokens)
    return dict(input_ids=torch.tensor([tokens + [specification['padTokenId']] * count]),
                attention_mask=torch.tensor([[1] * len(tokens) + [0] * count]), use_cache=False)


def run(model_dir, prompt, profile_name='qwen05b-k-projection', cancelled=None, mutate=None, progress=lambda step: None):
    profile = load_profile(profile_name)
    specification = profile.specification()
    if not specification or not isinstance(prompt, str) or not 0 < len(prompt.encode()) <= specification['maxPromptUtf8Bytes']:
        raise ValueError('Unsupported real-model prompt')
    for dependency, expected in POLICY['dependencies'].items():
        if version(dependency) != expected:
            raise ValueError('Pinned encrypted experiment dependency required')
    if json.loads(ext.default_params()) != POLICY['parameters']:
        raise ValueError('Unqualified cryptographic parameter change')
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError('Encrypted experiment cancelled')
    directory = Path(model_dir)
    validate_source(directory, specification)
    source_digests = {path.name: digest(path.read_bytes()) for path in sorted(Path(__file__).parent.glob('*.py'))}
    for name in ['policy.json', 'qwen05b.json']:
        source_digests[name] = digest(Path(__file__).with_name(name).read_bytes())
    torch.set_num_threads(2)
    began = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    private_tokens = tokenizer.encode(prompt)
    if not 1 <= len(private_tokens) <= specification['maxPromptTokens']:
        raise ValueError('Prompt exceeds the fixed private context bucket')
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, use_safetensors=True,
                                              trust_remote_code=False, torch_dtype=torch.float32, attn_implementation='eager').eval()
    name = specification['projection']['name']
    source = model.get_submodule(name)
    if (source.in_features, source.out_features) != (profile.input_width, profile.output_width):
        raise ValueError('Pinned projection geometry changed')
    selected = ClientLinearProjection(source)
    parent, leaf = name.rsplit('.', 1)
    setattr(model.get_submodule(parent), leaf, selected)
    with torch.no_grad():
        model(**fixed_inputs(list(range(specification['contextLength'])), specification))
    executor = GLWELinearLayerExecutor(use_dynamic_quantization=True)
    matrix = selected.prepare(executor, specification['quantizationBits'])
    weight_id = digest(matrix.tobytes(order='C'))
    if weight_id not in profile.expected_weight_ids:
        raise ValueError('Pinned quantized projection changed')
    model_id = digest(json.dumps(dict(profileId=profile.policy_id, source=specification['source'], weightId=weight_id), sort_keys=True).encode())
    executor.private_key, executor.compression_key = ext.cpu_create_private_key(executor.glwe_crypto_params)
    bridge = None
    try:
        bridge = EvaluatorBridge(model_id, executor.compression_key, {weight_id: matrix}, cancelled=cancelled, mutate=mutate, profile=profile)
        executor.fhext = SimpleNamespace(**{key: getattr(ext, key) for key in dir(ext) if not key.startswith('_')})
        executor.fhext.matrix_multiplication = bridge.matrix_multiplication
        setup_ms = (time.perf_counter() - began) * 1000
        initial_length = len(private_tokens)
        step_ms, float_ms, clear_ms, errors, quantization_errors = [], [], [], [], []
        with torch.no_grad():
            for step in range(profile.steps):
                inputs = fixed_inputs(private_tokens, specification)
                logits = {}
                for mode, measurements in [('float-reference', float_ms), ('clear-reference', clear_ms), ('execute', step_ms)]:
                    selected.mode = mode
                    started = time.perf_counter()
                    logits[mode] = model(**inputs).logits[0, len(private_tokens) - 1]
                    measurements.append((time.perf_counter() - started) * 1000)
                error = float((logits['execute'] - logits['clear-reference']).abs().max())
                quantization_error = float((logits['clear-reference'] - logits['float-reference']).abs().max())
                errors.append(error); quantization_errors.append(quantization_error)
                next_token = int(logits['execute'].argmax())
                if (error > specification['maxEncryptedVsQuantizedLogitError']
                        or next_token != int(logits['clear-reference'].argmax())
                        or specification['requireFloatGreedyEquality'] and next_token != int(logits['float-reference'].argmax())):
                    raise ValueError('Encrypted real-model generation disagrees with the pinned correctness contract')
                private_tokens.append(next_token)
                progress(step + 1)
        if bridge.calls != profile.steps * profile.projection_count:
            raise ValueError('Incomplete encrypted projection route')
        bridge.finish()
        return dict(schema='doppler.real-model-encrypted-projection/v1', qualified=False, productionBackendAvailable=False,
                    source=specification['source'], sourceDigests=source_digests, modelId=model_id, policyId=profile.policy_id,
                    keyContextId=bridge.key_id, evaluationBackend='cpu', projection=specification['projection'],
                    clientLocalOutputTokenIds=private_tokens[initial_length:], clientLocalOutputText=tokenizer.decode(private_tokens[initial_length:]),
                    matchesQuantizedGreedyReference=True, matchesFloatGreedyReference=True,
                    fullModelEncrypted=False, modelParameterCount=sum(value.numel() for value in model.parameters()), encryptedProjectionCalls=bridge.calls, steps=profile.steps,
                    setupMs=setup_ms, stepMs=step_ms, floatReferenceMs=float_ms, quantizedReferenceMs=clear_ms,
                    elapsedMs=(time.perf_counter() - began) * 1000, encryptedReplayMs=bridge.replay_seconds * 1000,
                    evaluatorRoundTripMs=bridge.remote_seconds * 1000, sentBytes=bridge.sent_bytes, receivedBytes=bridge.received_bytes,
                    publicEvaluationKeyBytes=len(bridge.key_bytes), maxEncryptedVsQuantizedLogitErrors=errors,
                    quantizationLogitErrors=quantization_errors, integrity=specification['integrity'], leakage=specification['leakage'],
                    clientPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                    providerPeakRssBytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                    environment=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(), dependencies=POLICY['dependencies']))
    finally:
        try:
            if bridge is not None:
                bridge.close()
        finally:
            executor.private_key = executor.compression_key = None
            executor.fhext = ext
            selected.executor = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    arguments = parser.parse_args()
    prompt = sys.stdin.buffer.read(load_profile('qwen05b-k-projection').specification()['maxPromptUtf8Bytes'] + 1).decode('utf-8')
    print(json.dumps(run(arguments.model_dir, prompt, progress=lambda step: print(f'Encrypted Qwen step {step} complete', file=sys.stderr, flush=True)), indent=2))
