# Frozen W8A8 model qualification

The independent CPU comparison fails the frozen application-quality gate. W8A8
answers 5/12 tasks correctly, F16 5/12 and the incumbent Q4 2/12. The gate requires
10/12 and preservation of every F16-passing task. W8A8 loses `literal` while gaining
`copy-code`; all three complete every task with EOS. These failures are retained.

The separate likelihood criterion passes: over identical 200 teacher-forced
targets, W8A8 mean NLL is 4.717551487921449, F16 is 4.630961223030469 and Q4 is
5.1531543895274305. W8A8's increase over F16 is 0.08659026489098043, below the frozen
0.10 limit. This approximation limit does not replace distributed tolerance 0.001.

`source-comparison.json` binds all inputs and compressed reports. Decompress each
`.json.gz` normally; the index records the SHA-256 of the uncompressed bytes.
No standard package, model catalog, numerical fixture or deployment was changed.

## Independent computation

`tools/gemma3-deployed-generation-control.py` loads every declared deployed tensor
and checks the manifest, tokenizer and shard identities. The pinned original
Gemma source configuration/tokenizer supplies independent Transformers equations.
Arithmetic outside quantized projections is F32 and attention caches store F16.

For W8A8, `tools/lib/w8a8_model_reference.py` replaces exactly the 126 explicitly
encoded linear matrices with dynamic per-token activation quantization, exact
integer-code GEMM and two separately rounded F32 rescale operations with explicit
subnormal-result flushing. Integer GEMM uses binary64 BLAS: products and all
partial sums are integers smaller than 2^31, so no integer accuracy is lost.
This is a numerical oracle, not an integer CPU speed claim. Every replacement's
execution count is retained. It imports no converter or runtime arithmetic.

The independent BigInt reference cross-check covers random exponent scales, a
near-halfway input where intermediate F32 division produces the wrong code,
integer accumulation above 2^24, signed-zero flushing and rejected inputs.
Native WGSL floating operations do not universally guarantee nearest-even;
GPU equivalence requires the separately tested explicit rounding implementation.

The existing F16 artifact differs from nearest-even conversion of the identified
BF16 source only in 18,887 values in the F16 subnormal region, with maximum
5.960464477539063e-8. The actual incumbent remains the comparison baseline; neither
its bytes nor historical results were replaced. Both original lineage and
expected-F16 rounding audits are retained.

## Executed scope

- `node tests/kernels/w8a8-reference.test.js`: independent rational reference.
- `python3 tests/integration/w8a8-model-reference-test.py`: three cross-reference controls.
- Existing Gemma deployed-weight decoding test: five controls.
- Three CPU full-model evaluations with four threads each; every task and teacher-forced target retained.

GPU installed-model, complete lifecycle and measured performance are separate
pending checks. Passing operator tests does not close the failed model-quality gate.
