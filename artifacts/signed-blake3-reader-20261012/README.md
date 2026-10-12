# Signed BLAKE3 source acquisition

Component: `doppler.runtime-source.client`. Intent preserved. Boundary: Rig's
qualified BLAKE3 manifest and SHA256 Capsule artifacts/pieces enter Run.

The actual signed candidate failed before acquisition because Run required
SHA256 manifest shards. The focused regression retained the same failure.
The repair accepts the two supported manifest hash algorithms and retains exact
signed manifest/index binding, path/size/weight membership, complete coverage,
and SHA256 verification of every acquired piece. SHA256 whole-shard equality
is still enforced when the manifest uses that algorithm. Rig independently
checks complete BLAKE3 shard identity before signing; Run does not claim to
verify that whole-shard hash from a selected range.

`verify-actual.js` verifies the actual v3 signature and reads only the first
64-byte piece. The original default acquisition limit also rejected the index's
16 MiB maximum; that failure is retained. `loading.json` declares the existing
16 MiB host allowance explicitly, without changing runtime defaults.

Validation: both algorithms, corrupt bytes, unsupported/malformed hashes,
wrong manifest identity, incomplete coverage and size mismatch; existing signed
resident factory/lifecycle controls; strict TypeScript, source style and
architecture checks. Independent read-only review confirmed this integrity scope.
No GPU, physical opening, whole-model quality or public promotion is established.
