"""Read diagnostic tensors from the exact bytes identified by their receipts."""
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


def read_tensor(record, directory=None):
    file = Path(record['file'])
    if not file.is_absolute():
        if directory is None:
            raise ValueError(f'Relative tensor path needs a receipt directory: {file}')
        file = Path(directory) / file
    expected = record.get('sha256')
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f'Missing SHA-256 for {file}')
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Tensor SHA-256 mismatch: {file}')
    shape = record.get('shape')
    if not isinstance(shape, list) or not shape or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError(f'Invalid tensor shape: {file}')
    if file.suffix == '.npy':
        values = np.load(io.BytesIO(raw), allow_pickle=False)
        if values.dtype.kind != 'f' or values.dtype.itemsize != 4:
            raise ValueError(f'Expected F32 tensor: {file}')
    elif file.suffix in ('.f32', '.u32'):
        if len(raw) != math.prod(shape) * 4:
            raise ValueError(f'Tensor byte count disagrees with shape: {file}')
        values = np.frombuffer(raw, dtype='<f4' if file.suffix == '.f32' else '<u4').reshape(shape)
    else:
        raise ValueError(f'Unsupported diagnostic tensor encoding: {file}')
    if tuple(values.shape) != tuple(shape):
        raise ValueError(f'Tensor shape mismatch: {file}')
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite diagnostic tensor: {file}')
    return values


def load_verified_trace(path, key_fields):
    path = Path(path)
    raw = path.read_bytes()
    trace = json.loads(raw)
    records = trace.get('tensors')
    if not isinstance(records, list):
        raise ValueError(f'Trace has no tensor inventory: {path}')
    index = {}
    verified_bytes = 0
    for record in records:
        key = tuple(record[field] for field in key_fields)
        if key in index:
            raise ValueError(f'Duplicate diagnostic tensor {key}: {path}')
        values = read_tensor(record, path.parent)
        verified_bytes += values.nbytes
        index[key] = record
    proof = dict(traceSha256=hashlib.sha256(raw).hexdigest(), tensors=len(index),
                 verifiedTensorBytes=verified_bytes)
    return trace, index, proof
