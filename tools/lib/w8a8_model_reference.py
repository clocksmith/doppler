"""Offline W8A8 model arithmetic; never imported by Doppler's runtime.

Binary64 division decides nearest integer codes exactly for binary32 operands:
with |x/s| <= 127, a non-tie is separated from any half-integer by at
least 1/(2*(2**24-1)); binary64 error is smaller. Exact ties stay exact.
Integer-code GEMM uses binary64 BLAS: every product and every partial sum is
an integer with magnitude <= 133144*127**2 < 2**31, hence exactly represented.
This is a correctness oracle, not an integer CPU performance measurement.
"""
import numpy as np
import torch

MIN_NORMAL = np.float32(2.0 ** -126)
MAX_K = 133144


def quantize_rows(values):
    values = np.asarray(values)
    if values.dtype != np.float32 or values.ndim != 2 or not 0 < values.shape[1] <= MAX_K:
        raise ValueError('Expected bounded F32 rows')
    magnitude = np.abs(values)
    if not np.isfinite(values).all() or ((magnitude > 0) & (magnitude < MIN_NORMAL)).any():
        raise ValueError('W8A8 input must be finite normal or zero')
    maxima = magnitude.max(axis=1)
    scales = (maxima.astype(np.float64) / 127).astype(np.float32)
    scales[maxima == 0] = 1
    if (scales < MIN_NORMAL).any():
        raise ValueError('W8A8 scale must be normal')
    codes = np.rint(values.astype(np.float64) / scales[:, None].astype(np.float64))
    return np.clip(codes, -127, 127).astype(np.int8), scales


def decode_rows(data, entry):
    n, k = entry['shape']
    padded = (k + 3) // 4 * 4
    if not 0 < k <= MAX_K or n < 1 or entry['layout'] != 'row':
        raise ValueError('Invalid W8A8 shape/layout')
    expected = {'packing': 'w8a8-row-v1', 'blockShape': [1, k], 'blockBytes': padded + 4,
                'paddedK': padded, 'rowStrideBytes': padded + 4, 'scaleOffsetBytes': 0, 'dataOffsetBytes': 4}
    if entry['storage'] != expected or len(data) != n * (padded + 4) or len(data) != entry['size']:
        raise ValueError('W8A8 storage differs from its shape')
    rows = np.frombuffer(data, np.uint8).reshape(n, padded + 4)
    scales = rows[:, :4].copy().view('<f4').reshape(n)
    codes = rows[:, 4:].copy().view(np.int8)
    if not np.isfinite(scales).all() or (scales < MIN_NORMAL).any():
        raise ValueError('Invalid stored W8A8 scale')
    if (codes == -128).any() or (codes[:, k:] != 0).any():
        raise ValueError('Invalid stored W8A8 code/tail')
    codes = codes[:, :k].copy()
    if ((codes == 0).all(axis=1) & (scales != 1)).any():
        raise ValueError('Zero W8A8 row must have unit scale')
    return codes, scales


def rescale(accumulators, activation_scales, weight_scales):
    value = accumulators.to(torch.float32) * activation_scales[:, None]
    if not torch.isfinite(value).all():
        raise ValueError('Activation rescale overflow')
    value = torch.where(value.abs() < float(MIN_NORMAL), torch.copysign(torch.zeros_like(value), value), value)
    value = value * weight_scales[None, :]
    if not torch.isfinite(value).all():
        raise ValueError('Weight rescale overflow')
    return torch.where(value.abs() < float(MIN_NORMAL), torch.copysign(torch.zeros_like(value), value), value)


class IndependentW8A8Linear(torch.nn.Module):
    def __init__(self, codes, scales):
        super().__init__()
        self.register_buffer('codes', torch.from_numpy(codes.copy()).double().t().contiguous())
        self.register_buffer('scales', torch.from_numpy(scales.copy()))
        self.calls = 0
        self.rows = 0

    def forward(self, values):
        shape = values.shape
        codes, scales = quantize_rows(values.detach().reshape(-1, shape[-1]).numpy())
        exact = torch.from_numpy(codes).double() @ self.codes
        if (exact.abs() > 2147483647).any() or not torch.equal(exact, exact.round()):
            raise ValueError('Integer accumulation bound violated')
        self.calls += 1
        self.rows += len(codes)
        output = rescale(exact.to(torch.int32), torch.from_numpy(scales), self.scales)
        return output.reshape(*shape[:-1], self.scales.numel())


def install_w8a8_modules(model, encoded):
    installed = {}
    for name, (codes, scales) in encoded.items():
        if not name.endswith('.weight'):
            raise ValueError('W8A8 tensor is not an explicit linear weight')
        module_name = name[:-len('.weight')]
        original = model.get_submodule(module_name)
        if not isinstance(original, torch.nn.Linear) or original.bias is not None:
            raise ValueError('W8A8 reference requires a bias-free source linear')
        parent_name, child = module_name.rsplit('.', 1)
        replacement = IndependentW8A8Linear(codes, scales)
        setattr(model.get_submodule(parent_name), child, replacement)
        installed[name] = replacement
    return installed
