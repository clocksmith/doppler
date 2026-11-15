"""Actual encrypted tiny-GPT2 hybrid generation; quarantined CPU qualification experiment."""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
from concrete.ml.torch.compile import build_quantized_module
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
from concrete.ml.common.utils import HybridFHEMode
import concrete_ml_extensions as ext
from bridge import EvaluatorBridge
from protocol import POLICY, POLICY_ID, digest


class PaddedProjection(nn.Module):
    """Explicit public zero-input padding; extra outputs are discarded on the client."""
    def __init__(self, source):
        super().__init__()
        self.inputs, self.outputs = source.weight.shape
        width = POLICY['matrixWidth']
        if max(self.inputs, self.outputs) > width:
            raise ValueError('Unsupported fixture geometry')
        self.remote = nn.Linear(width, width)
        self.mode, self.calibration, self.quantized, self.executor = 'calibrate', None, None, None
        with torch.no_grad():
            # Symmetric unused public coefficients avoid the library's asymmetric-weight rejection.
            # They multiply padded zeros or produce discarded outputs, preserving the original map.
            pattern = torch.arange(width * width).reshape(width, width) % 2 * 2 - 1
            self.remote.weight.copy_(pattern * source.weight.abs().max())
            self.remote.bias.zero_()
            self.remote.weight[:self.outputs, :self.inputs] = source.weight.T
            self.remote.bias[:self.outputs] = source.bias

    def forward(self, value):
        padded = torch.nn.functional.pad(value, (0, POLICY['matrixWidth'] - self.inputs))
        if self.mode == 'calibrate':
            self.calibration = padded.detach().clone()
            return self.remote(padded)[..., :self.outputs]
        modes = {'clear-reference': HybridFHEMode.DISABLE, 'execute': HybridFHEMode.EXECUTE}
        if self.mode not in modes or self.quantized is None or self.executor is None:
            raise ValueError('Unprepared encrypted projection')
        return self.executor.forward(padded.detach(), self.quantized, modes[self.mode])[..., :self.outputs]


def fixed_inputs(tokens):
    if not 1 <= len(tokens) <= POLICY['contextLength']:
        raise ValueError('Unsupported private context length')
    padding = POLICY['contextLength'] - len(tokens)
    return dict(input_ids=torch.tensor([tokens + [50256] * padding]),
                attention_mask=torch.tensor([[1] * len(tokens) + [0] * padding]), use_cache=False)


def run(model_dir, seed=15496, mutate=None, cancelled=None, progress=lambda step: None):
    for dependency, expected in POLICY['dependencies'].items():
        if version(dependency) != expected:
            raise ValueError('Pinned encrypted experiment dependency required')
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 50257:
        raise ValueError('Unsupported fixture seed')
    if json.loads(ext.default_params()) != POLICY['parameters']:
        raise ValueError('Unqualified cryptographic parameter change')
    model_dir = Path(model_dir)
    if (digest((model_dir / 'model.safetensors').read_bytes()) != POLICY['source']['safetensorsSha256']
            or digest((model_dir / 'config.json').read_bytes()) != POLICY['source']['configSha256']):
        raise ValueError('Pinned model fixture required')
    source_digests = {path.name: digest(path.read_bytes()) for path in sorted(Path(__file__).parent.glob('*.py'))}
    source_digests['policy.json'] = digest(Path(__file__).with_name('policy.json').read_bytes())
    torch.set_num_threads(2)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).eval()
    if (model.config.n_embd, model.config.n_layer, model.config.n_head, model.config.vocab_size) != (2, 2, 2, 50257):
        raise ValueError('Unsupported fixture model')
    original = copy.deepcopy(model)
    projections, padding_error = [], 0.0
    for name, module in list(model.named_modules()):
        if isinstance(module, Conv1D):
            projection = PaddedProjection(module)
            public_probe = torch.linspace(-1, 1, POLICY['contextLength'] * projection.inputs).reshape(1, POLICY['contextLength'], projection.inputs)
            with torch.no_grad():
                error = float((projection(public_probe) - module(public_probe)).abs().max())
            padding_error = max(padding_error, error)
            parent, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), leaf, projection)
            projections.append(projection)
    if len(projections) != POLICY['projectionCount'] or padding_error > 1e-6:
        raise ValueError('Public padded circuit differs from the model')
    # Explicit lower-level GLWE adapter: no mutation of HybridFHEModel's size heuristics.
    # Calibration uses fixed public fixture tokens, never the private request seed.
    with torch.no_grad():
        model(**fixed_inputs([0, 1, 2, 3]))
    executor = GLWELinearLayerExecutor(use_dynamic_quantization=True)
    for projection in projections:
        projection.quantized = build_quantized_module(projection.remote, projection.calibration,
                                                     n_bits=POLICY['quantizationBits'], keep_onnx=False)
        projection.executor = executor
        projection.calibration = None
    matrices = {}
    for projection in projections:
        quantized = projection.quantized
        _, operation = next(iter(quantized.quant_layers_dict.values()))
        weight, _ = executor._extract_weight_params(operation, bool(operation.attrs.get('transB', 0)))
        quantized_weight, *_ = executor._per_channel_weight_quantization(weight, quantized, torch.device('cpu'))
        matrix = np.ascontiguousarray(quantized_weight.long().numpy().astype(np.uint64), dtype='<u8')
        matrices[digest(matrix.tobytes(order='C'))] = matrix
    if len(matrices) != POLICY['projectionCount']:
        raise ValueError('Unexpected public weight identities')
    model_id = digest(json.dumps(dict(source=POLICY['source'], policyId=POLICY_ID, matrices=sorted(matrices)), sort_keys=True).encode())
    executor.private_key, executor.compression_key = ext.cpu_create_private_key(executor.glwe_crypto_params)
    with_key_ms = (time.perf_counter() - started) * 1000
    bridge = EvaluatorBridge(model_id, executor.compression_key, matrices, mutate=mutate, cancelled=cancelled)
    try:
        # An instance proxy routes only encrypted matmul; upstream encryption and decryption remain client-owned.
        executor.fhext = SimpleNamespace(**{name: getattr(ext, name) for name in dir(ext) if not name.startswith('_')})
        executor.fhext.matrix_multiplication = bridge.matrix_multiplication
        private_tokens, quantized_tokens, float_tokens = [seed], [seed], [seed]
        step_ms, logit_errors, quantization_errors = [], [], []
        setup_ms = (time.perf_counter() - started) * 1000
        with torch.no_grad():
            for _ in range(POLICY['steps']):
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError('Encrypted experiment cancelled')
                began = time.perf_counter()
                for projection in projections:
                    projection.mode = 'execute'
                encrypted_logits = model(**fixed_inputs(private_tokens)).logits[0, len(private_tokens) - 1]
                step_ms.append((time.perf_counter() - began) * 1000)
                for projection in projections:
                    projection.mode = 'clear-reference'
                clear_logits = model(**fixed_inputs(private_tokens)).logits[0, len(private_tokens) - 1]
                float_logits = original(**fixed_inputs(private_tokens)).logits[0, len(private_tokens) - 1]
                logit_errors.append(float((encrypted_logits - clear_logits).abs().max()))
                quantization_errors.append(float((clear_logits - float_logits).abs().max()))
                private_tokens.append(int(encrypted_logits.argmax()))
                quantized_tokens.append(int(clear_logits.argmax()))
                float_tokens.append(int(float_logits.argmax()))
                if private_tokens != quantized_tokens or private_tokens != float_tokens or logit_errors[-1] > POLICY['maxLogitError']:
                    raise ValueError('Encrypted transformer disagrees with fixture reference')
                progress(len(step_ms))
        if bridge.calls != POLICY['steps'] * POLICY['projectionCount']:
            raise ValueError('Incomplete encrypted transformer route')
        bridge.finish()
        return dict(schema=POLICY['schema'], qualified=False, productionBackendAvailable=False,
                    modelId=model_id, evaluationBackend=POLICY['evaluationBackend'], sourceDigests=source_digests, source=POLICY['source'], environment=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(), dependencies=POLICY['dependencies']),
                    clientLocalOutputTokenIds=private_tokens[1:], matchesQuantizedReference=True, matchesFloatGreedyReference=True,
                    steps=POLICY['steps'], encryptedProjectionCalls=bridge.calls, privateKeySent=False,
                    withKeyMs=with_key_ms, setupMs=setup_ms, stepMs=step_ms, elapsedMs=(time.perf_counter() - started) * 1000,
                    encryptedReplayMs=bridge.replay_seconds * 1000, evaluatorRoundTripMs=bridge.remote_seconds * 1000,
                    sentBytes=bridge.sent_bytes, receivedBytes=bridge.received_bytes, publicEvaluationKeyBytes=len(bridge.key_bytes),
                    maxLogitErrors=logit_errors, quantizationLogitErrors=quantization_errors, paddingMaxError=padding_error,
                    clientPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                    providerPeakRssBytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                    integrity='Full redundant encrypted replay before response decryption; no succinct proof or independent review',
                    leakage=POLICY['leakage'])
    finally:
        try:
            bridge.close()
        finally:
            executor.private_key = None
            executor.compression_key = None
            executor.fhext = ext
            for projection in projections:
                projection.executor = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--seed', type=int, default=15496)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.seed, progress=lambda step: print(f'Encrypted transformer step {step} complete', file=sys.stderr, flush=True)), indent=2))
