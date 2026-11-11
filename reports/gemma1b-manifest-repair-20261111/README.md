# Gemma 3 1B manifest repair

Publication and the catalog update are deferred by the user. No hosted model bytes or catalog pins changed.

The pinned HF manifest at `7c3d30e300bcb02cbd68fb0db3eee64fbf738f99` fails current validation: missing FFN branch mode, output scaling, residual scaling and nullable LongRoPE fields. It also has old kernel digests and historical BLAKE3 shard hashes. Runtime validation remains unchanged.

The original manifest, all 16 weight shards and the bundled tokenizer were authenticated against the immutable HF Git/LFS inventory. `tools/migrate-rdrr-shard-hashes.js` prepared SHA-256 metadata in a separate directory. `tools/refresh-converted-manifest.js` refreshed that candidate from the existing canonical Gemma 1B conversion configuration. The receipt records every inference delta, including attention offset 5 and current kernel digests. Weights and tokenizer are unchanged; execution metadata is changed, so old qualification is not transferred.

`node-smoke.json` records an Apple M3 Node/WebGPU run. Prompt: "Explain in one sentence why the sky is blue." Output: "The sky’s blue color comes from **electromagnetic radiation (light) scattering off of tiny particles in the atmosphere.**** \n\nLet me know if you'd like a more detailed explanation!" This establishes nonempty coherent generation, not factual or instruction-following quality. The user approved this output and the repaired artifact, then deferred HF authentication/publication.

Metadata is staged locally under `.cache/gemma1b-repair/` and retained here. Downloaded shard copies were removed after verification to recover disk space; their immutable URLs and SHA-256 hashes remain in the inventory and receipt. The exact replacement manifest is `manifest.json`. Publish only that file and the migration receipt, then verify the immutable remote revision before changing the catalog and generated registry. Do not upload weights or silently use this candidate for the current public pin.

`browser-projection.json` preserves an unsuccessful first harness attempt: its
candidate URL had a doubled slash and failed to resolve the tokenizer. After
correcting that harness URL, the successful installed-package browser check is
recorded in `../package-release-0.7.2/browser-projection.json`. The failed attempt
is not acceptance evidence.
