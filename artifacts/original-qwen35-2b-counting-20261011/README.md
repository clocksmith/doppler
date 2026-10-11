# Original Qwen3.5-2B weight control

The original pretrained checkpoint produces exactly the same **372 tokens**, with
EOS after counting to **120**, as the retained deployed-weight source control and
Doppler. The unchanged request asks for one through two hundred, one number per
line. The task still fails; no output, prompt, tolerance, stop token or limit was
altered to pass it.

This disproves the tested hypothesis that deployed Q4_K weight preparation is
necessary for this particular failure. Distribution and Doppler runtime execution
are also unnecessary to reproduce it. It does not establish behavior under other
prompts, generation settings, arithmetic, or cache policies, or general model
incapability.

The source is `Qwen/Qwen3.5-2B` pinned to revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. The 4,548,221,488-byte original
SafeTensors checkpoint matches its pinned upstream SHA256
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
The original BF16 text weights are converted exactly to F32 for the same independent
Transformers equations used by the deployed-weight control. The source's tied
embedding/output weight is restored; the separately quantized deployed LM head is
not loaded. Strict state loading checks all text tensor shapes and keys.

The 24 captured prompt IDs round-trip through the original tokenizer. Original
and deployed vocabularies agree despite different serialized tokenizer files.
The exact no-thinking template, greedy selection, full-history repetition penalty
`1.1`, 4,096-token allowance, 8,192-token context allowance and stop-token set
remain unchanged. F16 KV storage and F32 source arithmetic remain fixed to isolate
weight preparation. This is not a native BF16 execution comparison.

The control reuses `replay_generation` from the existing independent diagnostic;
the exact executed module is retained as `generation-tool.py.gz`. No runtime or
application source was changed. `source-result.json.gz` retains all inputs,
outputs, settings, dependency versions, source identities and code hashes.
`review.json` records the failed application outcome separately from completed
diagnosis. Original input and historical failures remain in
`artifacts/source-model-reference-20261011/`.

Reproduction (local source paths identify retained files, not bundled weights):

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
python3 tools/qwen35-original-weight-generation.py \
  --source /var/tmp/doppler-qwen35-2b-source-20261011 \
  --revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --deployed models/local/qwen-3-5-2b-q4k-ehaf16 \
  --control artifacts/source-model-reference-20261011/model-2b-counting-control.json.gz \
  --out /tmp/original-qwen35-2b-counting.json --threads 8
```

The source snapshot has no `generation_config.json`; an initial download inventory
request for that absent optional file stopped before model loading. The actual
control uses the required captured generation settings, not inferred source
defaults. Every necessary downloaded file's size/hash and immutable URL are in
`download-receipt.json`; upstream file metadata is retained in `source-index.json`.

Component: `doppler.repository-tooling`.
Intent: preserved.
Acceptance evidence: completed original-weight replay, strict source/tensor/token
identity checks, exact 372-token comparison, and Python syntax validation.
Boundary effects: offline attribution only; no runtime, package, reference or
application behavior changes, and no new model campaign.
