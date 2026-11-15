"""Trusted client projection orchestration; pinned Concrete ML owns quantization and encrypted arithmetic."""
import numpy as np
import torch
from torch import nn
from concrete.ml.common.utils import HybridFHEMode
from concrete.ml.torch.compile import build_quantized_module


class ClientLinearProjection(nn.Module):
    def __init__(self, linear, input_width=None, crop_width=None):
        super().__init__()
        self.remote = linear
        self.inputs = linear.in_features if input_width is None else input_width
        self.outputs = linear.out_features if crop_width is None else crop_width
        if not 0 < self.inputs <= linear.in_features or not 0 < self.outputs <= linear.out_features:
            raise ValueError('Invalid explicit projection geometry')
        self.mode, self.calibration, self.quantized, self.executor = 'calibrate', None, None, None

    def forward(self, value):
        if value.shape[-1] != self.inputs:
            raise ValueError('Client projection input geometry changed')
        padded = torch.nn.functional.pad(value, (0, self.remote.in_features - self.inputs))
        if self.mode == 'float-reference':
            return self.remote(padded)[..., :self.outputs]
        if self.mode == 'calibrate':
            self.calibration = padded.detach().clone()
            return self.remote(padded)[..., :self.outputs]
        modes = {'clear-reference': HybridFHEMode.DISABLE, 'execute': HybridFHEMode.EXECUTE}
        if self.mode not in modes or self.quantized is None or self.executor is None:
            raise ValueError('Unprepared encrypted projection')
        return self.executor.forward(padded.detach(), self.quantized, modes[self.mode])[..., :self.outputs]

    def prepare(self, executor, bits):
        if self.calibration is None:
            raise ValueError('Public calibration required before key generation')
        self.quantized = build_quantized_module(self.remote, self.calibration, n_bits=bits, keep_onnx=False)
        self.executor, self.calibration = executor, None
        _, operation = next(iter(self.quantized.quant_layers_dict.values()))
        weight, _ = executor._extract_weight_params(operation, bool(operation.attrs.get('transB', 0)))
        quantized_weight, *_ = executor._per_channel_weight_quantization(weight, self.quantized, torch.device('cpu'))
        return np.ascontiguousarray(quantized_weight.long().numpy().astype(np.uint64), dtype='<u8')
