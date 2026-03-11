# Translate Compare Student Promotion Checklist

Operational checklist for taking the EN/ES student from "artifact incoming" to
"usable in the Doppler translate compare console".

Use this alongside:
- [docs/developer-guides/05-promote-model-artifact.md](./developer-guides/05-promote-model-artifact.md)
- [docs/translate-compare-evidence-contract.md](./translate-compare-evidence-contract.md)

## Preconditions

Before touching catalog or demo-visible metadata, have all of these:
- final student checkpoint selected
- human-reviewed coherence result
- deterministic prompt/output notes
- frozen external metrics
- tokenizer/provenance confirmed

## Required Inputs

Need from Gamma:
- final student model ID to use in Doppler
- source artifact or converted runtime artifact
- tokenizer assets and any template requirements
- frozen evidence bundle fields:
  - `student.modelId`
  - `student.bleu`
  - `student.chrf`
  - `student.sizeBytes`
  - receipt links

## Doppler Steps

1. Convert or import the student artifact.
   - Keep the artifact reproducible.
   - Verify the exact artifact you intend to surface.

2. Run a real coherence pass before metadata changes.
   - Use deterministic translation settings.
   - Record the prompt and output you reviewed.

3. Add or sync the canonical support-registry entry first.
   - Do not treat `models/catalog.json` as the primary source of truth.

4. Sync the repo mirror and derived docs.
   - `models/catalog.json`
   - `docs/model-support-matrix.md` via sync tooling

5. Set the demo/student slot to the real model ID.
   - The compare console resolves the student lane from the evidence bundle or
     explicit global override.
   - Keep `student.modelId` equal to the actual demo-facing ID.

6. Publish the evidence bundle.
   - Match the contract in `docs/translate-compare-evidence-contract.md`.
   - Do not omit frozen fields; use `null` for intentionally blank values.

7. Re-run the browser demo with the actual student.
   - Verify `Translate -> Compare -> Proof preset`
   - Verify local history and share-link restoration
   - Verify the student lane resolves without manual lane edits

## Optional Follow-up

- Add a Transformers.js mapping only if there is a real public ONNX baseline
  that matches the claim you want to make.
- If no real mapping exists, keep engine-parity documented as unsupported for
  the TranslateGemma baseline.

## Final Verification

- `npm run support:matrix:sync` if catalog metadata changed
- `npm run ci:catalog:check`
- browser smoke of the compare console with the real student artifact
- human review of the exact public evidence values shown in the UI
