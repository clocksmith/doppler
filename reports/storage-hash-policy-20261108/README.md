# Local shard verification cost

`receipt.json` records six interleaved pairs on an Apple M3, Node 22.20.0,
Doppler `0ebdbe08`. Each operation reads the same 64 MiB synthetic file from a
primed OS page cache through `createNodeFileArtifactStorageContext`. One warmup
pair is retained separately. Output bytes are checked outside the timed region;
all fourteen reads matched. No GPU, network, OPFS or model inference ran.

The observed medians were 13.17 ms with explicit verification disabled and
40.49 ms with SHA-256 verification enabled. This is a storage-only observation
under ordinary development activity, not an estimate of complete model startup
or a cross-device performance claim. All samples are retained without trimming.

Reproduce from the repository with Node 22:

```
node tools/experiments/shard-hash-load.js /tmp/shard-hash-receipt.json
```

The runtime keeps verification enabled by default. Its corruption-rejection,
range verification, file/HTTP and policy tests are separate from this timing.
