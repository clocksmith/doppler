# Local signed-opening candidate

Rig preparation `ca9226fa` uses the exact frozen `5c2b3188` runtime/shader source,
archive `102fd0c4900c6744fb7d103a4b3609365b57e8538657bd47b8dfd8e7ac3469e7`,
the four physical resident reports, and original-source short greedy control.
No model, tokenizer, shader or captured numerical output is changed.

`capsule.json` is the direct Rig v2 output. The exact frozen runtime verified
its signature and all 54 artifact hashes. Its existing `migrateCapsuleV2` helper
produced `capsule-v3.json`, separating executable identity from the signed local
release history. That history expires after 24 hours and retains exclusions for
public chat, document quality and unobserved providers. It is not public promotion.
Only public signing keys are retained here; the private key stays outside Git.

The initially assumed subgroup target was rejected; `rejected-target.log` and
the original release preserve it. The selected portable target follows actual
manifest/kernel requirements; it does not claim universal device qualification.

Full model/artifact closure: `/var/tmp/qwen35-capsule-0.6.28-20261012`.
Physical signed opening remains pending. The frozen reader rejects BLAKE3 shard
metadata; the separately tested runtime repair is retained in
`../../signed-blake3-reader-20261012/`. A changed archive needs its own acceptance.
