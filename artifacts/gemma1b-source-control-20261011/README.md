# Independent replay of the rejected Gemma candidate

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Boundary effects: none; offline diagnostic only. Runtime, model artifacts,
application model selection and acceptance thresholds are unchanged.

The additional model evaluation rejected the existing Gemma 1B deployed artifact
on four of five fresh tasks. This control investigates two short failures using
Transformers' source equations, the exact deployed weights, identical captured
input tokens and the same generation policy. The source configuration is pinned
to the revision recorded in the artifact origin. Every shard and tokenizer is
SHA-256 checked against Reploid's retained diagnostic binding. The metadata-only
prepared manifest remains separately identified.

The duplicate-label request emits the same nine tokens, including EOS after
only four of six requested labels. The negation request emits the same three
tokens, including EOS after “Yes”, when permission is absent. Both failures thus
reproduce outside Doppler. This does not establish correctness for other tasks,
qualify the model or resolve the Qwen numerical comparison.

The independent gguf decoder accounts for Q4_K row padding. BF16 norm weights
are decoded exactly; source arithmetic is F32 with the deployed F16 cache
boundary. Five synthetic controls check F16/F32/BF16 values, distinct padded
Q4_K rows and rejected truncated storage. Full source outputs and inputs,
file identities, package versions and executed tool hashes are retained.

The initial invocation used an outdated Transformers import; the next input
omitted its required model identity. Both failures are retained. The corrected
input carries the identity from the original quality report, checked before
model allocation. No failed invocation is counted as inference evidence.

Reproduction (the model directory contains the identified prepared manifest
and original shard/tokenizer files):

```bash
PYTHONPATH=/tmp/doppler-reference-deps python3 tools/gemma3-deployed-generation-control.py \
  --model /path/to/identified-gemma-fixture \
  --model-identity sha256:f947e3ccac8028c9fcd0eb87cad3deec023102672288c75353d97d3b813e731b \
  --binding artifacts/gemma1b-source-control-20261011/binding.json \
  --source-config artifacts/gemma1b-source-control-20261011/source-config.json \
  --source-revision dcc83ea841ab6100d6b47a070329e1ba4cf78752 \
  --generation-control artifacts/gemma1b-source-control-20261011/repeated-labels-gemma-input.json \
  --out /tmp/gemma-repeated-labels-source.json
```

Use `negation-gemma-input.json` for the second case. Both completed with
`sameTokens: true`. Python compilation and the five decoding controls pass.
