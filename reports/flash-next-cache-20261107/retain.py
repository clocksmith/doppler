"""Retain small receipts and raw logits, without copying model or source trees.

Usage: python3 retain.py DESTINATION
Existing files must match, so repeating collection cannot overwrite evidence.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

target = Path(sys.argv[1])
sources = {
    'gpu-budget': '/tmp/doppler-flash-next-cache-20261107',
    'generic-host': '/tmp/doppler-flash-next-hostcore-20261107',
    'expert-all': '/tmp/doppler-flash-next-expert-host-20261107',
    'expert-repeat': '/tmp/doppler-flash-next-repeat-host-20261107',
}
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inventory = []
for name, directory in sources.items():
    destination = target / name
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(Path(directory).iterdir()):
        if not item.is_file() or item.name in ('package.json', 'package-lock.json'):
            continue
        if not (item.suffix in ('.json', '.log', '.f32', '.py', '.js') or item.name.endswith('.tar.gz')):
            continue
        copy = destination / item.name
        source_hash = digest(item)
        if copy.exists():
            assert digest(copy) == source_hash, f'Refusing to overwrite changed evidence: {copy}'
        else:
            shutil.copy2(item, copy)
        assert digest(copy) == source_hash
        inventory.append(dict(file=str(copy.relative_to(target)), bytes=copy.stat().st_size, sha256=source_hash))
(target / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
print(json.dumps(dict(files=len(inventory), bytes=sum(row['bytes'] for row in inventory))))
