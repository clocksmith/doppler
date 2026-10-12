# Actual Qwen 0.8B source topology for signed resident preparation

Component: Rig source intake and topology validation. Intent: preserved.
Boundary effects: incomplete homogeneous ModelIR v1 lowering now rejects
recurrent geometry and custom recurrent schedules. Frozen runtime archive
`102fd0c4…` is unchanged; this later source repair does not inherit its package
qualification.

The physical candidate describes its hybrid decoder using
`architecture.linearNumKeyHeads` and related geometry, plus
`inference.layerPattern.layerTypes`. Rig's v1 guard missed those representations.
With complete ordinary homogeneous input, the regression previously accepted
recurrent geometry and discarded it. The repaired guard rejects all five linear
geometry fields and a schedule containing `linear_attention`; callers must
supply the existing explicit ModelIR v2 path. No model-name detection or
runtime fallback was added.

`guard-before.log.gz` records the missing exception; `guard-after.log.gz` passes the
new cases and existing stage tests. `incomplete-test-setup.log.gz` is an earlier
invalid test setup which lacked normalized source artifacts; it is not the
reproducing evidence. `rig-regression.log.gz`, `style.log.gz`, `architecture.log.gz` and
`types.log.gz` retain the subsequent checks.

## Source intake

The existing `forgeSourceTruthFromFiles` from clean source `5c2b3188` generated
`model-ir-receipt.json` using `source-spec.json` and the explicit current repository
as source root. Its intake digest is
`sha256:6d4cb840ee764a839d6bc5da7537909dd8634a47cdbde5f27e7c6f23f78cc146`.

- Original checkpoint: `Qwen/Qwen3.5-0.8B`, revision
  `2fc06364715b967f1860aea9cf38778875588b17`.
- Original config bytes match the previously retained verified download receipt.
- The original safetensors file was rehashed on x@128 and matched that receipt.
  `source/header-binding.json` binds its exact initial header bytes, retained as
  `source/header.bin`. The JSON header summary is explanatory; intake reads and
  parses the pinned original binary header.
- The installed Transformers 5.6.2 implementation is retained as text with SHA
  `3e2b6239e4b2c3e512d4f9836a1dba12e975ae652407e84f2d3c2beebf0c9528`, matching
  the implementation behind the existing reviewed semantic index. The sigmoid
  output-gate fact additionally points to its actual attention expression.
- Unlike the earlier normalized 27B example, this intake does not invent a
  `language_model_only` configuration key. EOS comes from the actual text config;
  gate type comes from the implementation; all tensor geometry comes from the
  original 0.8B header.

The receipt preserves 24 text blocks, 12 vision blocks and the source's one
multi-token-prediction block. Text generation is the sole lowered entry point;
vision and speculative generation remain unlowered. **Qualified entry points
are empty.** This is source metadata, not physical execution or model quality.
The source's tied head and F32 recurrent-state semantics remain source facts;
the Q4 output-head policy and approved paired runtime representation are separate
implementation choices, not edits to those facts.

The first CLI attempt used the clean export's implicit repository root and
could not find the new source files (ENOENT before intake). Calling the same
existing API with the explicit source root produced this receipt; no validator
or frozen archive was changed to accommodate it.

`source-token-control.json` prepares the unchanged original first request
(19 input tokens, greedy policy, 1024-token allowance) for the existing original
weight evaluator. It binds the actual two-token physical result as a comparison,
not an independent expected answer. No successful source-token parity or signed
Capsule follows until the independently executed source result passes the
existing requirements. Document quality remains a separate failed/unresolved
application gate.

## Executed original-source token comparison

`source-token-result.json` retains the original checkpoint evaluator output.
The exact 19 input IDs round-trip through the deployed tokenizer; independent
original-weight execution generates `[9419, 248046]` (`Hello`, then EOS), matching
the frozen physical result without consuming runtime logits or cache state.
`source-token-parity.json` binds both results and records this bounded pass.
The evaluator's broader `qualified: false` remains unchanged: this is one short
source-token control, not document quality or whole-model qualification.

Execution used CPU cores 0–3 on x@128, four threads, and the retained Python
environment. The first invocation failed before inference because the existing
`blake3` dependency directory was absent from `PYTHONPATH`; its log is preserved.
Adding `/tmp/doppler-reference-deps` to the environment allowed the unchanged
control and evaluator to run. No dependency or model was downloaded, and no
GPU workload was launched. Both logs are retained as compressed original bytes.

## Program and piece preparation

`derive-generation-report.js` projects the actual first split request into Rig's
existing generation transcript. It binds source/control/capture/model identities,
checks every supplied token and sampled successor, and retains the source output
without rewriting it. Independent review confirmed the sampled-token assertion
runs before the numerical fixture advances its retained continuation. The source
evaluator independently advances its own argmax. Prompt parity covers original
tokenizer round-tripping of captured IDs; it does not claim template equivalence.

The first transcript projection omitted the explicit nullable seed required by
Program Bundle schema. `generation-report-before-schema.json` preserves it. The
corrected projection records `seed: null` for the observed greedy run and fills
the existing exact-count source-parity fields; it changes neither generation nor
the source result. Archived 0.6.28 tooling then builds the closed program with 46
artifacts. `program-bundle-preparation.json` binds this local output and the
verified 19-file, 537-piece index; model bytes stay outside Git.

Actual Rig construction then rejected the unchanged BLAKE3 manifest solely
because `loadCapsulePieceIndex` required SHA-256 shards. The repaired owner now
accepts the two already supported shard hash algorithms and verifies the complete
shard using its declared algorithm while retaining SHA-256 checks for each piece.
The regression also found that individually valid SHA-256 pieces could previously
be accepted with an incorrect whole-shard hash. Seven controls now cover both
algorithms, that mismatch, corrupted source pieces and unsupported algorithms.
The original physical-input rejection and failing regression are retained.

Signed tooling controls, source types, style and architecture checks pass. The
first type command named a nonexistent script, and the first correct type run
found a missing TypeScript narrowing; both failures remain alongside the final
passing log. These are preparation-source repairs, not changes to the frozen
runtime archive. Signed opening, native full qualification, demanding lifecycle
and document usefulness still require their own evidence.
