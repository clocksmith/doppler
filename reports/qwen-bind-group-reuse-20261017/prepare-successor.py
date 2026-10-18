"""Create a separate local qualification artifact; never mutate its predecessor."""
import copy
import hashlib
import importlib.metadata
import json
import pathlib
import sys
import time

import blake3

source, destination, report = map(pathlib.Path, sys.argv[1:])
source, destination, report = source.resolve(), destination.resolve(), report.resolve()
index_path = pathlib.Path('reports/weight-representation-20261017/model-2b-pieces.json')
expected_index = 'fb16dcc36771b478d811e83b3fad453cd1ee23b8b01d33eb0b797412032109dc'
expected_manifest = '83e5a01a21e234d238ea3ed5b0b234cc52781ddfce51861abd0fdf0ae3199028'
sha = lambda data: hashlib.sha256(data).hexdigest()
index_bytes = index_path.read_bytes()
manifest_bytes = (source / 'manifest.json').read_bytes()
assert sha(index_bytes) == expected_index, 'trusted inventory drift'
assert sha(manifest_bytes) == expected_manifest, 'predecessor identity drift'
index, manifest = json.loads(index_bytes), json.loads(manifest_bytes)
assert index['manifestIdentity'] == 'sha256:' + expected_manifest
assert destination != source and source not in destination.parents
assert not destination.exists(), 'refuse to overwrite an artifact'
expected = {s['filename'] for s in manifest['shards']} | {manifest['tokenizer']['file']}
assert {f['path'] for f in index['files']} == expected
destination.mkdir(parents=True)
report.mkdir(parents=True, exist_ok=True)
receipt = {'schema': 'doppler.successor-byte-verification/v1', 'passed': False,
    'source': str(source), 'destination': str(destination),
    'predecessorManifestSha256': expected_manifest, 'trustedPieceIndexSha256': expected_index,
    'trustSource': {'repository': 'clocksmith/reploid',
        'commit': 'dd5af6cfbac57ab8749b8306e6b7714d0724a9cd',
        'path': 'self/config/model-pieces/qwen-2b-pieces.json'},
    'independentBlake3': {'implementation': 'Python blake3',
        'version': importlib.metadata.version('blake3')},
    'files': [], 'releaseStatus': 'unsigned-local-qualification-candidate'}
started = time.monotonic()
try:
    digests = {}
    for entry in index['files']:
        name = entry['path']
        assert pathlib.Path(name).name == name, 'flat artifact paths only'
        path = source / name
        before = path.stat()
        data = path.read_bytes()
        assert len(data) == entry['size']
        cursor = 0
        for piece in entry['pieces']:
            assert piece['offset'] == cursor
            block = data[cursor:cursor + piece['size']]
            assert 'sha256:' + sha(block) == piece['identity'], name
            cursor += piece['size']
        assert cursor == len(data)
        standard = blake3.blake3(data).hexdigest()
        with (destination / name).open('xb') as output:
            output.write(data)
        # A separate file, not a hard link: later local edits cannot alter history.
        copied = (destination / name).read_bytes()
        assert sha(copied) == sha(data)
        assert blake3.blake3(copied).hexdigest() == standard
        assert path.stat().st_mtime_ns == before.st_mtime_ns
        digests[name] = standard
        receipt['files'].append({'file': name, 'bytes': len(data), 'sha256': sha(data),
            'blake3': standard, 'trustedPieces': len(entry['pieces'])})
        print('verified and copied', name, flush=True)
    successor = copy.deepcopy(manifest)
    for shard in successor['shards']:
        shard['hash'] = shard['blake3'] = digests[shard['filename']]
    successor['artifactIdentity']['manifestVariantId'] += '-standard-blake3-v1'
    successor['metadata']['integrityMigration'] = {
        'schema': 'doppler.standard-blake3-successor/v1',
        'predecessorManifestIdentity': 'sha256:' + expected_manifest,
        'trustedPieceIndexIdentity': 'sha256:' + expected_index,
        'weightAndTokenizerBytes': 'unchanged',
        'qualification': 'required',
    }
    new_bytes = (json.dumps(successor, indent=2) + '\n').encode()
    new_identity = sha(new_bytes)
    (destination / 'manifest.json').write_bytes(new_bytes)
    successor_index = copy.deepcopy(index)
    successor_index['manifestIdentity'] = 'sha256:' + new_identity
    index_output = (json.dumps(successor_index, indent=2) + '\n').encode()
    (destination / 'pieces.json').write_bytes(index_output)
    (report / 'predecessor-manifest.json').write_bytes(manifest_bytes)
    (report / 'successor-manifest.json').write_bytes(new_bytes)
    (report / 'successor-pieces.json').write_bytes(index_output)
    receipt['successorManifestSha256'] = new_identity
    receipt['successorPieceIndexSha256'] = sha(index_output)
    assert (source / 'manifest.json').read_bytes() == manifest_bytes
    unchanged = copy.deepcopy(successor)
    unchanged['shards'] = manifest['shards']
    unchanged['artifactIdentity'] = manifest['artifactIdentity']
    unchanged['metadata'] = manifest['metadata']
    assert unchanged == manifest, 'unexpected semantic change'
    receipt['passed'] = True
except Exception as error:
    receipt['failure'] = repr(error)
    raise
finally:
    receipt['elapsedSeconds'] = time.monotonic() - started
    (report / 'successor-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'passed': receipt['passed'], 'manifestSha256': new_identity,
    'files': len(receipt['files']), 'elapsedSeconds': receipt['elapsedSeconds']}))
