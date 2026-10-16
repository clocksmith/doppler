# Run from the repository root. Downloads the pinned public mirror, not a different checkpoint.
import hashlib, json, pathlib, urllib.request
root = pathlib.Path(__file__).parent
policy = json.loads((root / 'policy.json').read_text())
expected = json.loads((root / 'source-files.json').read_text())
expected['model.safetensors'] = policy['source']['weightsSha256']
destination = pathlib.Path('.cache/mlx-gemma270m-source')
destination.mkdir(parents=True, exist_ok=True)
for name, digest in expected.items():
    target = destination / name
    if target.exists():
        with target.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == digest:
                print(name, 'verified')
                continue
    source = policy['source']
    url = f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{name}"
    partial = target.with_suffix(target.suffix + '.download')
    checksum = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=30) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            checksum.update(chunk)
    if checksum.hexdigest() != digest:
        raise RuntimeError(f'{name}: pinned SHA256 mismatch; original target preserved')
    partial.replace(target)
    print(name, 'downloaded and verified')
