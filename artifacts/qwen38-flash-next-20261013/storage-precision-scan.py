"""Count BF16 values that F16 storage cannot represent exactly, per non-Q4K tensor class.
Usage: python3 -P storage-precision-scan.py <source-dir> > storage-precision-scan.json"""
import json, re, struct, sys
from pathlib import Path
import numpy as np
src = Path(sys.argv[1])
index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
SKIP = re.compile(r'^(mtp\.|model\.visual\.)|ngram_embedding\.shard_|mlp\.experts\.|ngram_heads_|layer_multipliers')
Q4K = re.compile(r'(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|shared_expert\.(gate|up|down)_proj)\.weight$')
def klass(n):
    n = re.sub(r'layers\.\d+\.', 'layers.N.', n)
    return n
stats = {}
headers = {}
for name, fn in sorted(index.items()):
    if SKIP.search(name) or Q4K.search(name):
        continue
    path = src / fn
    if fn not in headers:
        with open(path, 'rb') as h:
            n = struct.unpack('<Q', h.read(8))[0]
            headers[fn] = (8 + n, json.loads(h.read(n)))
    start, hdr = headers[fn]
    t = hdr[name]
    if t['dtype'] != 'BF16':
        continue
    b, e = t['data_offsets']
    with open(path, 'rb') as h:
        h.seek(start + b)
        raw = np.frombuffer(h.read(e - b), dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    f16 = f32.astype(np.float16)
    back = f16.astype(np.float32)
    inexact = int(np.count_nonzero(back != f32))
    overflow = int(np.count_nonzero(np.isinf(f16) & np.isfinite(f32)))
    flushed = int(np.count_nonzero((f16 == 0) & (f32 != 0)))
    absmax = float(np.max(np.abs(f32)))
    nz = np.abs(f32[f32 != 0])
    absmin = float(nz.min()) if nz.size else 0.0
    rel = float(np.max(np.abs(back[f32 != 0] - f32[f32 != 0]) / nz)) if nz.size else 0.0
    k = klass(name)
    s = stats.setdefault(k, {'tensors': 0, 'values': 0, 'inexact': 0, 'overflow': 0, 'flushed': 0, 'absmax': 0.0, 'absminNonzero': float('inf'), 'maxRelError': 0.0})
    s['tensors'] += 1; s['values'] += int(f32.size); s['inexact'] += inexact; s['overflow'] += overflow; s['flushed'] += flushed
    s['absmax'] = max(s['absmax'], absmax); s['absminNonzero'] = min(s['absminNonzero'], absmin); s['maxRelError'] = max(s['maxRelError'], rel)
print(json.dumps(stats, indent=1))
