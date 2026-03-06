# Doppler Lean Module

Purpose:
- formalize high-value execution-contract invariants for Doppler manifests
- catch incompatible manifest/session/execution-v0 combinations before runtime

Current scope:
- `Doppler/Model.lean`
  - core execution-contract vocabulary (`KVLayout`, `Phase`, `Dtype`, `OpClass`)
- `Doppler/ExecutionContract.lean`
  - compatibility predicates for KV layout vs attention phases
  - session-consistency predicates for BDPA and tiered layouts
  - proofs for the concrete bug class where a manifest declares a decode-only KV layout while also declaring prefill attention
- `Doppler/ExecutionContractFixtures.lean`
  - `TranslateGemma`-style conflicting and fixed fixtures
- `Doppler/Check.lean`
  - simple checker entry point that renders fixture verdicts

Initial target bug class:
- execution-v0 manifest contains attention steps with `phase = prefill` or `phase = both`
- session defaults choose `kvcache.layout = bdpa`
- runtime later fails because BDPA attention is decode-only

This module is intentionally narrow. It does not yet model:
- runtime merge algebra for JSON objects
- kernel-path alias resolution
- precision precedence and cast-step requirements
- manifest extraction into JSON artifacts

Manual check:

```bash
./lean/check.sh
```

Manifest-backed check:

```bash
node tools/lean-execution-contract.js \
  --manifest models/local/translategemma-4b-it-wq4k-ef16-hf16/manifest.json
```

The manifest-backed checker:
- extracts execution steps and session facts from `manifest.inference`
- resolves missing decode-loop fields through Doppler runtime defaults
- generates a temporary Lean module under `lean/.generated-*`
- compiles it against `Doppler/ExecutionContract.lean`

The same execution-contract class is now also enforced in JS manifest validation:
- [src/formats/rdrr/validation.js](/home/x/deco/doppler/src/formats/rdrr/validation.js)
- [src/config/execution-contract-check.js](/home/x/deco/doppler/src/config/execution-contract-check.js)

That means `parseManifest()` and manifest refresh flows fail fast on the same BDPA/prefill and session-default incompatibilities, even when Lean is not installed.

Expected output:

```text
["translategemma_conflicting_steps: fail", "translategemma_conflicting_session: fail",
 "translategemma_fixed_steps: pass", "translategemma_fixed_session: pass"]
lean-check: ok (leanprover/lean4:v4.16.0)
```

Example manifest-backed output:

```text
lean-execution-contract: wrote lean/.generated-XXXXXX/translategemma_4b_it_wq4k_ef16_hf16_ExecutionContractCheck.lean
"executionContractModule:translategemma_4b_it_wq4k_ef16_hf16_ExecutionContractCheck"
"executionContractOverall:pass"
["translategemma-4b-it-wq4k-ef16-hf16.steps: pass", "translategemma-4b-it-wq4k-ef16-hf16.session: pass"]
lean-execution-contract: ok (leanprover/lean4:v4.16.0)
```
