# Fetch the benchmark's pinned Hugging Face models; verify every LFS file against
# the tree's sha256 oid.
#   python3 -I fetch-models.py <destination root>
# Writes <root>/<repo id>/ (the layout Transformers.js reads) plus acquisition.json
# per repository.
import hashlib
import json
import os
import sys
import urllib.request

PINNED = [
    ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", []),
    ("onnx-community/kev-0.6b-ONNX", "5a827091b3d58fe4fdfc02a5a5b2ce364047d678", []),
    ("onnx-community/kev-4b-ONNX", "195bdf73096df2bca7150223c9279af7893163c4", []),
    ("onnx-community/open-jev-deberta-v3-large-ONNX", "7c79f25b5ac496089f448a969c801872ad59d31c", ["onnx/model.onnx"]),
]


def fetch(repo, rev, out, skip):
    tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"))
    record = []
    for f in tree:
        if f["type"] != "file" or any(f["path"].startswith(s) for s in skip):
            continue
        dest = os.path.join(out, f["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        oid = (f.get("lfs") or {}).get("oid")
        if not (os.path.exists(dest) and os.path.getsize(dest) == f["size"]):
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{rev}/{f['path']}") as r, open(dest + ".part", "wb") as w:
                while chunk := r.read(1 << 22):
                    w.write(chunk)
            os.replace(dest + ".part", dest)
        h = hashlib.sha256()
        with open(dest, "rb") as r:
            while chunk := r.read(1 << 22):
                h.update(chunk)
        digest = h.hexdigest()
        if oid and digest != oid:
            raise SystemExit(f"sha256 mismatch for {repo}/{f['path']}: {digest} != {oid}")
        if os.path.getsize(dest) != f["size"]:
            raise SystemExit(f"size mismatch for {repo}/{f['path']}")
        record.append({"path": f["path"], "size": f["size"], "sha256": digest, "lfsOidVerified": bool(oid)})
    with open(os.path.join(out, "acquisition.json"), "w") as w:
        json.dump({"repository": repo, "revision": rev, "files": record}, w, indent=1)
    print(repo, len(record), "files ok")


if __name__ == "__main__":
    root = sys.argv[1]
    for repo, rev, skip in PINNED:
        fetch(repo, rev, os.path.join(root, repo), skip)
