# Physical attention-cache verification

Component: `doppler.tests`, `doppler.repository-tooling`. Intent: preserved.
Boundary effects: none; the identified standard 0.6.27 archive and shaders are
unchanged. This is test-only instrumentation with an independent offline check.

The selected failing request (index 2, decode step 3) was replayed through its
four prefixes. Uninstrumented controls were collected first. The observer copies
actual K/V allocations returned to attention at each prefix; copies are queued
before the next step can write, mapped separately, and destroyed after readback.
Read identities, sequence lengths, layout, shape and byte ranges are recorded.
No captured value is fed into execution. Existing operator diagnostics retain
pre-storage K/RoPE and V/projection values for independent rounding comparison.

All **4,558,848 F16 values match bit-for-bit**, over six attention layers and
four prefixes (24 K/V snapshots). Independently applied NumPy F16 rounding
matches the physical stored bits exactly. Each subsequent step reads the same
allocation identities, with the expected history length and unchanged preceding
positions. Instrumented unmasked logits equal their baseline at every prefix.

This closes the storage, indexing and F32-to-F16 conversion hypothesis at the
exercised boundary. It does not validate independently evolved upstream values
or waive the full-model source mismatch. Historical comparisons still fail at
decode steps 2 and 3, with 0.001 unchanged. No arithmetic repair is justified by
this experiment, and no reference candidate is activated.

Validation: 21 synthetic source-diagnostic controls pass, including physical
bit mismatch, missing history, wrong position, wrong read identity and changed
instrumentation rejection. The mock GPU observer test verifies copy ordering,
mapping-failure cleanup and geometry rejection; it is separate from physical
execution. Syntax, CATSCAN and diff checks pass. Full raw physical results and
logs are retained compressed with hashes in `review.json` and `comparison.json`.

Reproduce the bounded physical capture:

```bash
node tests/integration/frozen-reference-producer-replay.js \
  artifacts/partition-contracts-20261010/doppler-gpu-0.6.27.tgz \
  models/local/qwen-3-5-0-8b-q4k-ehaf16 \
  /tmp/physical-cache.json current artifacts/physical-cache-20261011/target.json
```

Call `compare_physical_cache(capture, [3,7,11,15,19,23], 2, 256)` from
`tools/qwen35-deployed-reference.py` on the resulting JSON. This independently
checks physical storage against captured inputs; it is not a new acceptance
reference or a broad model-capture campaign.
