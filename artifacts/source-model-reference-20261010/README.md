# Independent deployed-weight numerical control

Component: Doppler development diagnostics. Intent: preserved.
Boundary effects: none; no Run arithmetic, package archive, accepted reference,
model bytes, or tolerance changed.

The offline Transformers 5.6.2 control uses source Qwen3.5 equations, independent
GGUF Q4_K decoding, the deployed F16 embedding and separately quantized LM head,
F32 arithmetic, and F16 KV storage. It consumes the captured prompt IDs and then
exact frozen-reference tokens, retaining the recurrent and attention state
between steps. This is a diagnostic, not a replacement reference or CPU fallback.

At the unchanged absolute tolerance 0.001, 34 of 55 comparisons with the current
Doppler capture fail (maximum 0.0020241737365722656). All 55 selected tokens agree.
The historical reference also differs from this control: 26 of 55 comparisons
fail (maximum 0.0018062591552734375). This does not justify reference activation.
The full comparison summaries and identities are in `summary.json`.

Source configuration: Qwen/Qwen3.5-0.8B at revision
`2fc06364715b967f1860aea9cf38778875588b17`, `config.json`, retained here as
`source-config.json`. Dependencies used are recorded in the summary. GGUF and
BLAKE3 were installed into `/tmp/doppler-reference-deps`, not the runtime package.

All 18 weight shards and tokenizer bytes were checked against the host-pinned
SHA-256 piece index. The manifest's legacy shard hash implementation is not
standard BLAKE3; the report records both values without treating them as equal.
The diagnostic used the index from Reploid `a05d1066`, SHA-256
`3358a1d398da150df99b1988481a2691e7ee4573c8019bfcfa8b2c5011075cd6`.
That index retained an older manifest binding. Reploid `b44f99a2` corrects only
that binding after independently checking all 277 pieces; the model bytes are
unchanged. Reproduce this diagnostic with its recorded index:

```bash
git -C ../reploid show a05d1066:self/config/model-pieces/qwen-0-8b-pieces.json > reports/local/source-model-reference-20261010/piece-index.json
PYTHONPATH=/tmp/doppler-reference-deps python3 tools/qwen35-deployed-reference.py \
  --model models/local/qwen-3-5-0-8b-q4k-ehaf16 \
  --source-config artifacts/source-model-reference-20261010/source-config.json \
  --capture reports/local/frozen-reference-producer/doppler-current026-full-linux.json.gz \
  --reference ../reploid/tests/fixtures/distributed-reference.json.gz \
  --piece-index reports/local/source-model-reference-20261010/piece-index.json \
  --piece-index-identity sha256:3358a1d398da150df99b1988481a2691e7ee4573c8019bfcfa8b2c5011075cd6 \
  --prefixes 55 --threads 8 \
  --out reports/local/source-model-reference-20261010/reproduced-prefixes.json
```

The complete local logits are retained at the summary's `rawResultPath`, with
its SHA-256. They are not included in this compact evidence bundle. Completion
of the script means the diagnostic ran; its exit status does not certify
numerical acceptance. `qualified` remains false.

Failed setup attempts and an invalid initial decode probe are retained locally
and identified in the summary. That invalid probe reset context for each decode
step; it was corrected before the reported 55-step result. Its large errors are
not used as model evidence.

Acceptance evidence: the command above (55 completed diagnostic steps),
`python3 -m py_compile tools/qwen35-deployed-reference.py`, and
`npm run catscan:check`. Further numerical work must isolate operation boundaries;
this control alone neither identifies a shader defect nor qualifies a reference.


## Retained prefill boundaries

`prefill-boundaries.json` compares independently computed source-model module
outputs with retained GPU tensors. Embeddings agree exactly. The first difference
is input normalization (maximum 7.152557373046875e-7); the first projection differs
by 1.1444091796875e-5. Differences accumulate through the layers. Final-normalized
activations differ by 0.0014710426330566406, and first-prefix logits differ by
0.0009584426879882812. These observations do not isolate a new defective kernel
or qualify the remaining decode prefixes.

The GPU capture uses an earlier manifest binding but has identical first-prefix
input IDs and output logits to the current capture; the tool asserts both before
comparison and records both identities. Hooks observe outputs and never replace
them. This CPU-only check does not rerun or interfere with a GPU campaign.

To reproduce, use the command above with `--prefixes 1`, the current piece index
from Reploid `b44f99a2` (SHA-256
`18adb1f08f4e694d357a27f7a5d67ea57a7d50c5e82aa7fda099084b445974b7`),
`--boundary-capture reports/local/frozen-reference-producer/doppler-current026-normalization-linux.json.gz`,
and a separate output path. The original 55-prefix tool version is retained in
Doppler `6faaee44`; its recorded tool hash and receipt remain unchanged.
