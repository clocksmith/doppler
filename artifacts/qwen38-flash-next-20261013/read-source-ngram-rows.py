"""Independently read n-gram table rows from the pinned checkpoint files.

Rows are located only from the authenticated header receipt (checkpoint-headers.json):
global row r lies in shard r // shardRows at file byte fileByteBegin + (r % shardRows) * rowBytes.
Compares the SHA-256 of the rows, in the address order recorded by validate-artifact-ngram-reads.js,
with the digest of the rows that path read through the converted artifact. Usage:
  python3 -P read-source-ngram-rows.py <source-dir> <artifact-reads.json>
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
table = json.loads((ROOT / 'checkpoint-headers.json').read_text())['ngramTable']
source = Path(sys.argv[1])
reads = json.loads(Path(sys.argv[2]).read_text())
shard_rows = table['shardRows'][0]
row_bytes = table['rowWidth'] * 2
shards = {s['shard']: s for s in table['shards']}
assert all(s['rowEnd'] - s['rowBegin'] == shard_rows for s in shards.values())
digest = hashlib.sha256()
handles = {}
for row in reads['addressList']:
    assert 0 <= row < table['rows'], row
    shard = shards[row // shard_rows]
    handle = handles.get(shard['file']) or handles.setdefault(shard['file'], open(source / shard['file'], 'rb'))
    handle.seek(shard['fileByteBegin'] + (row % shard_rows) * row_bytes)
    data = handle.read(row_bytes)
    assert len(data) == row_bytes
    digest.update(data)
print(json.dumps({'rows': len(reads['addressList']), 'sourceRowsSha256': digest.hexdigest(),
                  'artifactRowsSha256': reads['rowsSha256'], 'equal': digest.hexdigest() == reads['rowsSha256'],
                  'filesTouched': len(handles)}, indent=1))
