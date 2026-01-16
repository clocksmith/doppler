# Known-Good Matrix

This matrix defines the smallest set of fixtures that must load and produce
stable outputs. Keep it narrow and deterministic.

## Fixtures

| Fixture | Format | Quant | Notes | Test |
| --- | --- | --- | --- | --- |
| `tests/fixtures/mini-model` | RDRR | F32 | Small 2-layer transformer, bundled tokenizer | `tests/correctness/known-good-fixtures.spec.js` |
| `tests/fixtures/tiny-model` | RDRR | F32 | Alternate shape/layout for loader coverage | `tests/correctness/known-good-fixtures.spec.js` |
| `tests/fixtures/sample.gguf` | GGUF | F32 | Parser coverage only (no inference) | `tests/unit/formats-gguf.test.js` |

## Output Checksums

Known-good outputs are stored in:

```
tests/fixtures/known-good-outputs.json
```

To update:

```
DOPPLER_UPDATE_KNOWN_GOOD=1 npx playwright test -c tests/correctness/playwright.config.js
```
