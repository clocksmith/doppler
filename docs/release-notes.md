# Release Notes

## 2026-03-06

- Added hosted registry validation for local `models/catalog.json` entries with `lifecycle.availability.hf=true`.
- Added remote validation for demo-visible registry entries so the live demo does not surface non-fetchable hosted models.
- Added `npm run ci:catalog:check` to catch registry script drift, support-matrix drift, and HF catalog issues in one step.
- Added a repeatable Hugging Face publication workflow via `npm run registry:publish:hf -- --model-id <model-id>`.
