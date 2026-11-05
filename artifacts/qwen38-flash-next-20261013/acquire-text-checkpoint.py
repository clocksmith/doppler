"""Acquire the text-bearing checkpoint files of Qwen/Qwen3.8-Flash-Next at the pinned revision.

Downloads every safetensors file that holds at least one non-vision tensor (vision-only files are
skipped; MTP tensors share files with text tensors and are excluded later by conversion rules).
Each file is resumed with HTTP ranges, then verified: byte length and SHA-256 must equal the
Hugging Face LFS X-Linked-Size and X-Linked-ETag at the pinned revision, and its header bytes must
match the authenticated header receipt (checkpoint-headers.json). Writes acquisition-receipt.json in
the target directory. Usage:
  python3 -I acquire-text-checkpoint.py <target-dir> [--parallel 4]
"""
import concurrent.futures
import hashlib
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
headers = json.loads((ROOT / 'checkpoint-headers.json').read_text())
weight_map = json.loads((ROOT / 'source/model.safetensors.index.json').read_text())['weight_map']
assert headers['revision'] == binding['revision']
BASE = f"https://huggingface.co/{binding['model']}/resolve/{binding['revision']}/"
CHUNK = 8 << 20

def is_text(name):
    return 'visual' not in name

def linked(name):
    for attempt in range(8):
        try:
            return linked_once(name)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:  # transient network or DNS failure
            print(json.dumps({'file': name, 'identityRetry': attempt, 'error': str(error)[:200]}), flush=True)
            time.sleep(min(60, 2 ** attempt))
    return linked_once(name)

def linked_once(name):
    request = urllib.request.Request(BASE + name, method='HEAD', headers={'User-Agent': 'doppler-source-preparation'})
    opener = urllib.request.build_opener(type('NoRedirect', (urllib.request.HTTPRedirectHandler,), {
        'redirect_request': lambda self, *args, **kwargs: None})())
    try:
        response = opener.open(request, timeout=60)
        hdrs = response.headers
    except urllib.error.HTTPError as error:  # 302 to the CDN carries the LFS headers
        hdrs = error.headers
    size, etag = hdrs.get('X-Linked-Size'), (hdrs.get('X-Linked-ETag') or '').strip('"')
    assert size and len(etag) == 64, f'{name}: missing LFS identity headers'
    return int(size), etag

def fetch(name, target):
    size, etag = linked(name)
    path = target / name
    partial = path.with_suffix(path.suffix + '.part')
    if path.exists() and path.stat().st_size == size:
        source = path
    else:
        for attempt in range(8):
            have = partial.stat().st_size if partial.exists() else 0
            if have >= size:
                break
            request = urllib.request.Request(BASE + name, headers={'Range': f'bytes={have}-', 'User-Agent': 'doppler-source-preparation'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, open(partial, 'ab') as out:
                    assert response.status in (200, 206), (name, response.status)
                    if response.status == 200 and have:
                        out.truncate(0)
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
            except Exception as error:  # resume on the next attempt
                print(json.dumps({'file': name, 'retry': attempt, 'error': str(error)[:200]}), flush=True)
                time.sleep(min(60, 2 ** attempt))
        assert partial.stat().st_size == size, f'{name}: incomplete after retries'
        source = partial
    digest = hashlib.sha256()
    with open(source, 'rb') as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
        handle.seek(0)
        length = struct.unpack('<Q', handle.read(8))[0]
        header = handle.read(length)
    if digest.hexdigest() != etag:
        source.unlink()
        raise AssertionError(f'{name}: sha256 mismatch; removed for re-download')
    assert length == headers['files'][name]['headerBytes'] and hashlib.sha256(header).hexdigest() == headers['files'][name]['headerSha256'], \
        f'{name}: header differs from the authenticated receipt'
    if source != path:
        source.rename(path)
    return {'file': name, 'bytes': size, 'sha256': etag}

def main():
    target = Path(sys.argv[1]).resolve()
    parallel = int(sys.argv[sys.argv.index('--parallel') + 1]) if '--parallel' in sys.argv else 4
    target.mkdir(parents=True, exist_ok=True)
    files = sorted({f for name, f in weight_map.items() if is_text(name)})
    skipped = sorted(set(weight_map.values()) - set(files))
    started = time.time()
    done = []
    with concurrent.futures.ThreadPoolExecutor(parallel) as pool:
        for result in pool.map(lambda name: fetch(name, target), files):
            done.append(result)
            print(json.dumps({'verified': result['file'], 'count': len(done), 'of': len(files),
                              'elapsedSeconds': round(time.time() - started)}), flush=True)
    receipt = {'schema': 'doppler.qwen4-exp-text-acquisition/v1', 'model': binding['model'], 'revision': binding['revision'],
               'verification': 'byte length and SHA-256 equal HF X-Linked-Size / X-Linked-ETag; header bytes equal checkpoint-headers.json',
               'files': done, 'skippedVisionOnlyFiles': skipped, 'totalBytes': sum(f['bytes'] for f in done),
               'elapsedSeconds': round(time.time() - started)}
    (target / 'acquisition-receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'files': len(done), 'skipped': skipped, 'totalBytes': receipt['totalBytes']}))

if __name__ == '__main__':
    main()
