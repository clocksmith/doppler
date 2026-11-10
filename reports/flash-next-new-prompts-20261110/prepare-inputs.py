"""Freeze explicitly specified new prompts before independent source execution.

Usage: prepare-inputs.py SPEC ARTIFACT NEW_OUTPUT_DIRECTORY
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys

import tokenizers
from tokenizers import Tokenizer

spec_path, artifact, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
raw = spec_path.read_bytes()
spec = json.loads(raw)
manifest_raw = (artifact / 'manifest.json').read_bytes()
tokenizer_raw = (artifact / 'tokenizer.json').read_bytes()
require(sha(manifest_raw) == spec['manifestSha256'], 'Artifact manifest mismatch')
require(sha(tokenizer_raw) == spec['tokenizerSha256'], 'Tokenizer identity mismatch')
tokenizer = Tokenizer.from_str(tokenizer_raw.decode())
rows = []
for case in spec['cases']:
    tokens = tokenizer.encode(case['text'], add_special_tokens=False).ids
    require(case['minimumTokens'] <= len(tokens) <= case['maximumTokens'], 'Prompt length outside declared bounds: ' + case['id'])
    rows.append(dict(id=case['id'], text=case['text'], tokens=tokens))
require(rows and len({r['id'] for r in rows}) == len(rows), 'Empty or duplicate prompt cases')
output.mkdir()
payload = dict(schema='doppler.qwen38-learned-forward-inputs/v1', tokenizers=tokenizers.__version__, sequences=rows)
(output / 'inputs.json').write_text(json.dumps(payload, indent=2) + '\n')
receipt = dict(schema='doppler.flash-next-new-prompt-inputs/v1',
               createdAtUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               scope='Two new numerical parity prompts, not used to tune the preceding runtime repairs; no task-quality qualification',
               specSha256=sha(raw), producerSha256=sha(Path(__file__).read_bytes()),
               inputsSha256=sha((output / 'inputs.json').read_bytes()), manifestSha256=sha(manifest_raw),
               tokenizerSha256=sha(tokenizer_raw), inputs=[dict(id=r['id'], tokens=len(r['tokens'])) for r in rows],
               sourceRunsStarted=0, gpuRunsStarted=0, acceptanceChanged=False, normalModelQualified=False)
(output / 'input-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
