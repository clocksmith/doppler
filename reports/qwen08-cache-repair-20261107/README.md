# Qwen 3.5 0.8B persistent-cache repair

The original pinned manifest used historical nonstandard BLAKE3 shard digests.
The downloaded bytes were correct. Shard 0's SHA-256 matches the pinned Git-LFS
inventory; both current Doppler and independent BLAKE3 compute `11af394f…`, rather
than the old manifest's `4dd461be…`. Runtime verification was not weakened.

The existing `tools/migrate-rdrr-shard-hashes.js` authenticated the manifest,
tokenizer and all 18 shards against the immutable HF inventory, then regenerated
standard SHA-256 shard digests and derived artifact/integrity identities. Shard
and tokenizer bytes are unchanged. Original metadata remains at revision
`80d7716270b6371d541de979eff3370edaf34e13`.

Only the new manifest and migration receipt were published at
[ea7b211b72c4d43687fa72802dee7aba74eca293](https://huggingface.co/clocksmith/rdrr/commit/ea7b211b72c4d43687fa72802dee7aba74eca293).
The catalog and generated quickstart registry now pin it. The new manifest digest
is `c5fcf38e945f1c048faca9ccda8991f1494563317f46a6dd79e0714ad001c569`.

[Browser receipt](browser-cache-receipt.json): physical Apple M3 WebGPU imported
and verified all shards into OPFS, generated `Hello!`, then unloaded, reused the
verified cache and generated the same exact token IDs. This is a bounded text
smoke, not transferred full-model qualification. The manifest's partial vision
encoder remains unavailable with an explicit warning.

[Migration receipt](migration-receipt.json), [immutable inventory](source-lfs-inventory.json),
and [publication](publication.json) retain original/new identities and evidence.
