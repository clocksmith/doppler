# Translate Compare Evidence Contract

Canonical contract for the evidence bundle consumed by the demo translate
compare console.

Schema file:
- `demo/translate-compare-evidence.schema.json`

Sample payload:
- `demo/fixtures/translate-compare-evidence.sample.json`

## Purpose

This payload fills the proof strip and evidence snapshot in the existing
`Translate -> Compare` demo.

It is intentionally narrow:
- teacher metrics
- student metrics
- size comparison
- optional receipt links

It is not a general benchmark schema and it is not a training scoreboard
replacement.

## Delivery paths

The demo loads evidence from one of these sources:

1. `globalThis.__DOPPLER_TRANSLATE_COMPARE_EVIDENCE__`
2. `globalThis.__DOPPLER_TRANSLATE_COMPARE_EVIDENCE_URL`
3. built-in fallback placeholder

If no evidence is provided, the demo shows placeholder text and leaves the
numeric fields blank.

For local preview, point `__DOPPLER_TRANSLATE_COMPARE_EVIDENCE_URL` at the
sample fixture or inject the same JSON object through
`__DOPPLER_TRANSLATE_COMPARE_EVIDENCE__`.

## Required shape

Required top-level fields:
- `summary`
- `caution`
- `teacher`
- `student`
- `receipts`

Required model fields for both `teacher` and `student`:
- `label`
- `modelId`
- `bleu`
- `chrf`
- `sizeBytes`

The demo currently reads these values directly:
- `teacher.bleu`
- `teacher.chrf`
- `teacher.sizeBytes`
- `teacher.modelId`
- `student.bleu`
- `student.chrf`
- `student.sizeBytes`
- `student.modelId`
- `receipts[0].label`
- `receipts[0].href`
- `updatedAt`
- `summary`
- `caution`

## Example

```json
{
  "schemaVersion": 1,
  "updatedAt": "2026-03-11",
  "summary": "Compact EN/ES student remains close enough to the teacher to matter in-browser.",
  "caution": "Result is specific to EN/ES translation and should not be generalized to all language pairs.",
  "teacher": {
    "label": "Teacher",
    "modelId": "translategemma-4b-it-q4k-ehf16-af32",
    "bleu": 34.0474,
    "chrf": 62.18,
    "sizeBytes": 3167327178
  },
  "student": {
    "label": "Student",
    "modelId": "translategemma-en-es-student-q4k-ehf16-af32",
    "bleu": 32.4485,
    "chrf": 60.71,
    "sizeBytes": 892341221
  },
  "receipts": [
    {
      "label": "scoreboard.md",
      "href": "https://example.invalid/scoreboard.md"
    },
    {
      "label": "leaderboard.md",
      "href": "https://example.invalid/leaderboard.md"
    }
  ]
}
```

## Notes For Gamma

- Use the final chosen student only.
- Keep `student.modelId` equal to the actual Doppler/demo model ID you want the
  compare lane to resolve.
- If a metric is not frozen yet, send `null` rather than omitting the field.
- `sizeBytes` should be the downloadable artifact size used in the public proof,
  not a rough label.

## Non-goals

This contract does not try to carry:
- per-example samples
- full scoreboard rows
- run-history entries
- engine-specific runtime timings

Those belong in separate artifacts or receipt links.
