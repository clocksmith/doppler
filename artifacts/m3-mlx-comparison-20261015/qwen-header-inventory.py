# Read headers only; never download model payloads.
import concurrent.futures, hashlib, json, pathlib, struct, urllib.request
root = pathlib.Path(__file__).parent
source = {'repo': 'Qwen/Qwen3.8-27B', 'revision': '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'}
index = json.loads((root / 'qwen27b-source-index.json').read_text())
def fetch(name):
    url = f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{name}"
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-1048575'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise RuntimeError(f'{name}: server did not honor bounded range request')
        data = response.read(1048576)
    size = struct.unpack('<Q', data[:8])[0]
    if size + 8 > len(data):
        raise RuntimeError(f'{name}: header exceeds bounded request')
    header = data[8:8 + size]
    return {'file': name, 'headerSha256': hashlib.sha256(header).hexdigest(), 'headerBytes': size, 'tensors': json.loads(header)}
files = sorted(set(index['weight_map'].values()))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(fetch, files))
(root / 'qwen27b-source-headers.json').write_text(json.dumps({'source': source, 'shards': results}, indent=2) + '\n')
print('Read', len(results), 'headers;', sum(r['headerBytes'] for r in results), 'header bytes')
