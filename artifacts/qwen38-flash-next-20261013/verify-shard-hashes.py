# Recompute SHA-256 of every shard file named in an RDRR manifest and compare with the manifest.
import hashlib, json, os, sys, time
root, out = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(root, "manifest.json")))
assert m["hashAlgorithm"] == "sha256", m["hashAlgorithm"]
rows, bad, t0, total = [], [], time.time(), 0
for s in m["shards"]:
    p = os.path.join(root, s["filename"]); h = hashlib.sha256(); n = 0
    with open(p, "rb", buffering=0) as f:
        while chunk := f.read(16 << 20): h.update(chunk); n += len(chunk)
    ok = h.hexdigest() == s["hash"] and n == s["size"]
    total += n
    if not ok: bad.append({"index": s["index"], "filename": s["filename"], "size": n, "expectedSize": s["size"], "sha256": h.hexdigest(), "expected": s["hash"]})
extra = sorted(set(os.listdir(root)) - {s["filename"] for s in m["shards"]} - {"manifest.json"})
res = {"schema": "doppler.rdrr-shard-hash-verification/v1", "manifestSha256": hashlib.sha256(open(os.path.join(root, "manifest.json"), "rb").read()).hexdigest(),
       "shards": len(m["shards"]), "bytes": total, "mismatches": bad, "unlistedFiles": extra, "seconds": round(time.time() - t0, 1)}
json.dump(res, open(out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items()}))
