# Local issue triage

A small browser application that suggests whether a GitHub issue is a **bug**, a **feature**
request or a **question**, using typed choice scoring on this device. It never applies a label:
the maintainer accepts or overrides each suggestion, and every decision is recorded with the
maintainer's name. Suggestions with a small score margin are flagged for careful review.

Only the scoring engine is Doppler's. The application owns the routes, the prompt, body
truncation, the review threshold and the decision log ([triage.js](triage.js)).

## How it decides

For each route the app asks Qwen3-Reranker-0.6B, in its published yes/no prompt format, whether
the issue is that type, through `scoreChoices` with the single-token labels `yes` and `no`. The
route score is `logit(yes) − logit(no)`; the highest score is the suggestion. Scores are raw logits,
not probabilities.

## Files

| File | Role |
|---|---|
| [triage.js](triage.js) | Routes, prompt, truncation retry, suggestion and decision record |
| [engines.js](engines.js) | Doppler adapter (signed Capsule installed into OPFS) and a Transformers.js adapter with the same contract |
| [installation.js](installation.js) | Persists and verifies Capsule artifacts so later openings work offline |
| [index.html](index.html), [app.js](app.js) | The maintainer-facing page |
| [dataset.js](dataset.js) | Pinned NLBSE'24 evaluation data (fetched from upstream, never vendored) |
| [tools/](tools/) | Screening, latency/profile probes and the complete-application comparison |

## Run

The page expects `node_modules/doppler-gpu` (install the Doppler archive into this directory) and
a `descriptor.json` naming the Capsule URL, storage identifier and the signers and target plans
you trust. Serve the directory over HTTP from `localhost` or HTTPS; WebGPU and OPFS need a secure
context. A release descriptor is not yet published: the measured Capsule is test-signed.

## Measured

[Complete-application comparison](../../reports/issue-triage-comparison-20261022/README.md) on a
Radeon 8060S: 648 ms median per decision (Transformers.js 236 ms), 14 MB installed package
(690 MB), 2.0 GB peak browser memory (7.4 GB), and offline reopening plus crash recovery with
identical outputs, which the Transformers.js configuration could not do with this model. Zero-shot
accuracy on 150 held-out issues is 0.57 raw and 0.61 with label-free route centering; a trained
classifier reaches about 0.83 F1 on this dataset. Treat suggestions as triage help, not labels.

## Data

Evaluation uses the [NLBSE'24 issue report classification](https://github.com/nlbse2024/issue-report-classification)
data at revision `2927bc67`. Its repository's license file is empty, so issue text is fetched from
upstream at evaluation time and only indices, labels and scores are kept.
