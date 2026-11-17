"""Bounded temporary storage for exact integer reference products; one global dequantization."""
import json
from pathlib import Path
import numpy as np
import torch
from full_plan import ranges
from protocol import require_active

POLICY=json.loads(Path(__file__).with_name('reference-policy.json').read_text())


def integer_dot(inputs,weights,*,cancelled=None):
    require_active(cancelled)
    values=inputs.to(torch.int64)
    if values.ndim!=2 or weights.ndim!=2 or values.shape[1]!=weights.shape[0]:raise ValueError('Explicit integer reference geometry required')
    output=torch.zeros((values.shape[0],weights.shape[1]),dtype=torch.int64)
    for out_begin,out_end in ranges(weights.shape[1],POLICY['outputTileWidth']):
        for begin,end in ranges(weights.shape[0],POLICY['inputTileWidth']):
            require_active(cancelled)
            block=torch.from_numpy(np.array(weights[begin:end,out_begin:out_end],dtype=np.int64))
            output[:,out_begin:out_end]+=torch.matmul(values[:,begin:end],block)
    require_active(cancelled)
    return output
