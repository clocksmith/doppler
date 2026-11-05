"""Token inputs for prepare-learned-forward-reference.py, encoded with the pinned tokenizer.json.

short: a plain factual prompt. long: the first 2304 tokens of the pinned modeling_qwen4_exp.py, so
query positions 2051 and later exercise QSA sparse selection (dense limit budget + compression - 1 = 2051).
Usage: python3 -I make-learned-forward-inputs.py
"""
import hashlib
import json
from pathlib import Path
import tokenizers

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
files = {e['path']: e['sha256'] for e in binding['files'] + binding['referenceImplementation']['files']}
for name in ('tokenizer.json', 'modeling_qwen4_exp.py'):
    assert hashlib.sha256((ROOT / 'source' / name).read_bytes()).hexdigest() == files[name], name
tok = tokenizers.Tokenizer.from_file(str(ROOT / 'source/tokenizer.json'))
short_text = 'The quick brown fox jumps over the lazy dog. The capital of France is'
long_ids = tok.encode((ROOT / 'source/modeling_qwen4_exp.py').read_text(), add_special_tokens=False).ids[:2304]
out = {'schema': 'doppler.qwen38-learned-forward-inputs/v1', 'tokenizers': tokenizers.__version__,
       'sequences': [{'id': 'short', 'text': short_text, 'tokens': tok.encode(short_text, add_special_tokens=False).ids},
                     {'id': 'long', 'text': 'source/modeling_qwen4_exp.py, first 2304 tokens', 'tokens': long_ids}]}
assert len(long_ids) == 2304
(ROOT / 'learned-forward-inputs.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
print({s['id']: len(s['tokens']) for s in out['sequences']})
