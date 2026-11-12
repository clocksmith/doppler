"""Freeze the pinned prompt set's tokens before any CPU or GPU execution (procedure step 2).

Usage: python3 -P freeze-inputs.py PROMPT_SET.json ARTIFACT_DIR FREEZE_RECORD.json NEW_OUTPUT_DIRECTORY

Tokenizes each case's raw text with the artifact tokenizer, no special tokens, checks the
declared token bounds, and writes inputs.json plus input-receipt.json. The receipt binds the
committed freeze record (policy hash and freeze time) and records that no run has started.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys

import tokenizers
from tokenizers import Tokenizer

if len(sys.argv) != 5:
    raise SystemExit(__doc__)
prompt_set_path, artifact, freeze_path, output = map(Path, sys.argv[1:])
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


prompt_set_raw = prompt_set_path.read_bytes()
prompt_set = json.loads(prompt_set_raw)
freeze = json.loads(freeze_path.read_bytes())
require(prompt_set['schema'] == 'doppler.routing-aware-prompt-set/v1', 'Not a routing-aware prompt set')
require(freeze['promptSetFileSha256'] == sha(prompt_set_raw), 'Prompt set differs from the frozen one')
manifest_raw = (artifact / 'manifest.json').read_bytes()
require(sha(manifest_raw) == freeze['artifactManifestSha256'], 'Artifact differs from the frozen one')
tokenizer_raw = (artifact / 'tokenizer.json').read_bytes()
tokenizer = Tokenizer.from_str(tokenizer_raw.decode())
rows = []
for case in prompt_set['cases']:
    tokens = tokenizer.encode(case['text'], add_special_tokens=False).ids
    require(case['minimumTokens'] <= len(tokens) <= case['maximumTokens'], 'Token count outside declared bounds: ' + case['id'])
    rows.append(dict(id=case['id'], text=case['text'], tokens=tokens))
require(len({r['id'] for r in rows}) == len(rows) == 22, 'Expected 22 distinct pinned cases')
output.mkdir()
payload = dict(schema='doppler.qwen38-learned-forward-inputs/v1', tokenizers=tokenizers.__version__, sequences=rows)
(output / 'inputs.json').write_text(json.dumps(payload, indent=2) + '\n')
receipt = dict(schema='doppler.routing-aware-fresh-inputs/v1', createdAtUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               scope='Frozen pinned prompt tokens for the routing-aware fresh test; written before any CPU or GPU run',
               freezeRecordSha256=sha(freeze_path.read_bytes()), policySha256=freeze['policySha256'],
               promptSetSha256=freeze['promptSetSha256'], producerSha256=sha(Path(__file__).read_bytes()),
               inputsSha256=sha((output / 'inputs.json').read_bytes()), manifestSha256=sha(manifest_raw),
               tokenizerSha256=sha(tokenizer_raw), inputs=[dict(id=r['id'], tokens=len(r['tokens'])) for r in rows],
               cpuRunsStarted=0, gpuRunsStarted=0)
(output / 'input-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt['inputs']))
