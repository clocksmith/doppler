# Flash-Next: bounded independent decode comparison

The repaired frozen AMD Node F32-KV snapshot passes the unchanged
`doppler.boundary-tolerance/quantized-v1` check for the original short and long
prompts, including two cached decode steps per prompt. This is not general model
qualification, a current-main package check, or a deployment.

Each comparison covers all 248,320 last-position logits. All six greedy token
choices match the independent source. The six vectors contain 1,489,920 checked
values, with zero violations of `abs(actual - reference) <= 0.05 + 0.05 * abs(reference)`.

| Prompt | Stage | Max absolute error | RMS error | Violations | Greedy token |
| --- | --- | ---: | ---: | ---: | ---: |
| Short, 15 prompt tokens | Prefill | 0.0000286102 | 0.0000042194 | 0 | 11751 |
| Short | Decode 1 | 0.0000352859 | 0.0000062017 | 0 | 13 |
| Short | Decode 2 | 0.0000478029 | 0.0000097750 | 0 | 9664 |
| Long, 2,304 prompt tokens | Prefill | 0.0001218319 | 0.0000139240 | 0 | 7920 |
| Long | Decode 1 | 0.0001373291 | 0.0000196529 | 0 | 22257 |
| Long | Decode 2 | **0.0623848438** | **0.0112129281** | 0 | 2312 |

The final long decode has substantially greater numerical variation. Its maximum
absolute error alone is greater than 0.05; the unchanged policy also includes the
reference-relative term. The passing result does not establish bit-exact outputs
or small absolute error everywhere. Full values and precision are in
[comparison.json](comparison.json).

## Reference and input independence

The reference uses the pinned source implementation on CPU with float32 state,
the same converted Q4_K/F16 weights, and the source BF16 n-gram rows. Every decode
reference recomputes the complete extended prefix without a source KV cache;
Doppler performs cached decode. This tests agreement between those two paths,
not agreement with a separately cached source execution.

The original F32 source selected the first continuation token. An existing
source F16-cache control supplied a proposed second token. Neither came from
Doppler output. The new F32 source forwards then verified both continuation
choices **before the GPU run was prepared**. A mismatch would stop preparation.
[inputs-receipt.json](inputs-receipt.json) records that ordering and the source
identities; [protocol.json](protocol.json) binds the verified chains, reference
payloads, run and unchanged policy. [reference-inputs.json](reference-inputs.json)
contains the exact extended prefixes.

The source producer is unchanged (`prepare-learned-forward-reference.py`). It
uses pinned source model revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
and Transformers revision `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`, Torch
2.11.0+cu130 on CPU, 16 threads and highest float32 matmul precision.

## Execution and identity

The same loaded model runs short then long, with sequence state reset between
them. The long prefill payload is byte-identical to the previous passing
long-only run. The run completes with empty GPU-error and cleanup-failure lists.
The 27B provider remains active; no service action, GPU memory-limit change,
runtime change, precision change, catalog promotion or deployment belongs to
this work.

The tested snapshot is the repaired F32-KV snapshot from
[the route-product repair](../flash-next-route-products-20261110/README.md).
The three targeted production shaders match committed source. Other files in
that snapshot differ from current main, so this result does **not** qualify the
whole current checkout. The final [receipt.json](receipt.json) binds observed
source hashes, committed shader identities, the comparator, preparation tools,
policy, inputs and reference metadata. Large raw reference arrays and GPU
observations remain in the retained qualification volume, indexed by SHA-256.

## Comparator verification

The existing learned-model comparator now accepts an explicit F32 decode
protocol while preserving its historical F16 invocation. It checks complete
vectors, finite numbers, payload hashes, exact generated prefixes, capture
coverage and captured-vector argmax. A mismatched generated prefix produces a
failed, noncomparable row rather than comparing different contexts. Existing
output files are never overwritten. Identity checks also remain active under
Python `-O`.

Focused checks:

```sh
python3 tests/integration/flash-next-learned-comparison-test.py
python3 tests/integration/flash-next-trace-integrity-test.py
git diff --check
```

The comparator suite has 10 passing tests, including finite corruption, missing
steps, wrong prefixes, invalid hashes under normal and optimized Python,
nonfinite values, mismatched greedy choices, lane/counterfactual rejection,
immutable output and historical F16 compatibility. The trace-integrity suite
has seven passing tests. The retained historical F16 observations still fail
when checked by the updated comparator; see
[retained-f16-failure.json](retained-f16-failure.json). That is a replay of old
evidence, not a fresh run of the repaired F16 lane.

## Remaining qualification

- Different prompts and longer generation, including QSA block-boundary decode.
- Installed package and intended release configuration on this device.
- The original F16-KV lane and other platforms, qualified separately.
- Task quality, ordinary serving lifecycle and throughput, with their own scope.

The tested decode positions extend the 2,304-token prefix to 2,305 and 2,306;
they do not complete the next four-token QSA block. No general release decision
follows from this bounded check.

Component: `doppler.tests`, `doppler.repository-tooling`. Intent: preserved.
Boundary effects: reference preparation and evidence validation only; no runtime
or acceptance-policy change.
