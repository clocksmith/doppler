# Audit: Hardcoded Defaults and Config Bypass in Runtime Code

## Violation Class

**Config-bypass via hardcoded literal** — JavaScript code that returns a hardcoded
value for a behavior-changing tunable instead of reading from the resolved config
chain (manifest → preset → schema default → rule asset).

This audit covers the general case. Model-type guards that short-circuit the
preset/merge chain are the most common instance, but not the only one.

## Normative Rules

### 1. No Runtime Defaults in Code

`docs/style/general-style-guide.md` §"No Runtime Defaults in Code":

> Runtime code should read resolved config values directly. Do not add literal
> fallbacks for tunables in JS; put defaults in schemas and merge them in the
> config layer.
>
> For all behavior-changing choices (kernel selection, precision mode, fallback
> variants), the only fallback source is explicit config/rule assets. If policy
> is not present, raise a typed configuration error instead of silently selecting
> an alternate behavior.

### 2. Manifest-First Contract

`docs/style/javascript-style-guide.md` §"Manifest-First Contract":

> Any new inference knob must be wired end-to-end:
> - Add to `ManifestInferenceSchema` (and converter defaults if needed)
> - Populate in converter mapping (preset + HF config)
> - Merge in `src/config/merge.js` with `_sources`
>
> Do not reintroduce runtime model detection or preset fallbacks.

### 3. Config Merge Order

`docs/style/general-style-guide.md` §"Config Merge Order":

> `manifestInferenceConfig = merge(manifestDefaultConfig, modelPresetConfig, converterOverrideConfig, artifactDerivedConfig)`

Model-specific values belong in model presets (`src/config/presets/models/*.json`)
or schema defaults (`src/config/schema/`), not in JS functions.

### 4. Explicit over Implicit

`docs/style/general-style-guide.md` §"Explicit over Implicit":

> No magic, document everything.

Hardcoded literals buried in conditional branches are implicit. Config/preset
values are explicit and auditable.

### 5. Runtime Configuration (Performance Invariants)

`docs/style/javascript-style-guide.md` §"Runtime Configuration":

> Runtime code must respect dtype and performance invariants from config and
> device capabilities.
> - Do not hardcode `f32` fallbacks when `shader-f16` is available.
> - If a `f32` path is required, require an explicit config flag and log once
>   per session.

## What To Search For

Scan all `.js` files under `src/` for code that:

1. **Returns a hardcoded literal for a behavior-changing tunable** — `return true`,
   `return false`, `return <number>`, `return '<string>'` for a value that should
   come from config, a preset, a schema default, or a rule asset.

2. **Gates on model identity instead of reading config** — Checks `model_type`,
   `modelType`, `architectureHint`, or calls a helper like `isQwen*`, `isGemma*`,
   `isLlama*`, `isLfm*`, etc. to decide a config value.

3. **Provides a silent fallback instead of failing fast** — Falls back to a
   literal when the config field is missing/undefined, instead of throwing a
   configuration error.

4. **Hardcodes dtype, precision, or kernel selection** — Inline `'f16'`, `'f32'`,
   or kernel variant strings in conditional branches instead of reading from
   config or rule assets.

5. **Duplicates a value that exists in a preset or schema** — JS code that
   returns the same literal a preset already specifies. Even if not contradictory,
   this is a maintenance hazard and style violation.

### Concrete grep patterns

```bash
# Model-type identity checks that gate a hardcoded return
rg -n 'if\s*\(is[A-Z][a-zA-Z]*Config\(' src/ --glob '*.js'
rg -n 'model_type.*===' src/ --glob '*.js'
rg -n 'modelType\s*===' src/ --glob '*.js'
rg -n 'architectureHint\s*===' src/ --glob '*.js'

# Literal returns inside detect/resolve/build functions
rg -n -A5 'function detect|function resolve|function build' src/converter/manifest-inference.js

# Hardcoded dtype literals in runtime code
rg -n "return 'f16'|return 'f32'|return 'bf16'" src/ --glob '*.js' --glob '!*.test.js'

# Literal fallbacks for config tunables
rg -n 'return true|return false' src/converter/manifest-inference.js
```

## What Is NOT a Violation

- **Normalizing/canonicalizing values** (e.g., `normalizeCustomLayerType` mapping
  `'gated_delta'` → `'linear_attention'`). Type transformations are not config
  overrides.
- **Reading source checkpoint fields** (e.g., `modelConfig.rms_norm_eps`). Passing
  through HF config fields is artifact-derived config, a valid merge layer.
- **Detecting structural features from tensors** (e.g., `detectNormalizationFromTensors`
  checking tensor name patterns). Artifact-derived inference is not model-type gating.
- **Fail-fast validation** (e.g., `throw new Error('unknown model type')`). Errors
  are not silent overrides.
- **Preset selection** (e.g., mapping `architectureHint` → `presetId` in
  `conversion-plan.js`). Selecting which preset to load is correct use of model
  identity.
- **Constants that are not tunables** (e.g., `WORKGROUP_SIZE = 256`, mathematical
  constants, format magic bytes). These are structural invariants, not config.
- **Rule asset lookups** (e.g., `selectRuleValue('inference', 'dtype', ...)`).
  Rule-based selection is the correct mechanism for conditional behavior.

## Audit Procedure

For each finding:

1. **Identify the function** — name, file, line range.
2. **Identify the hardcoded value** — what literal is returned or assigned.
3. **Identify the gate** — what condition triggers the hardcoded path (model-type
   check, missing field, dtype comparison, etc.).
4. **Check the config chain** — does a preset, schema default, or rule asset
   already specify this field? If yes, is the JS value consistent or contradictory?
5. **Classify severity:**
   - **CRITICAL** — Hardcoded value contradicts what the config chain provides,
     causing silent wrong behavior.
   - **HIGH** — Hardcoded value fires before the config chain is consulted.
     Config chain may not specify the field, so the hardcoded value is the only
     source — but it should be moved to a preset or schema default.
   - **MEDIUM** — Hardcoded value fires after the config chain as a fallback,
     but the value should live in a preset or schema default, not in JS.
   - **LOW** — Hardcoded value is redundant with what the config chain already
     provides (no behavioral difference, but violates the style rule).
6. **Propose fix** — move the value to a preset/schema/rule asset, or remove
   the guard if the config chain already has the correct value.

## Known Violations (seed findings)

### 1. `detectRmsNormWeightOffset` — CRITICAL

**File:** `src/converter/manifest-inference.js:275-278`

```javascript
function detectRmsNormWeightOffset(presetInference, modelConfig, defaults) {
  if (isQwen35LinearAttentionConfig(modelConfig)) {
    return true;  // ← hardcoded, fires BEFORE preset check
  }
  if (typeof presetInference?.normalization?.rmsNormWeightOffset === 'boolean') {
    return presetInference.normalization.rmsNormWeightOffset;
  }
  return defaults.normalization.rmsNormWeightOffset;
}
```

**Gate:** `isQwen35LinearAttentionConfig` (model-type identity check).
**Hardcoded value:** `true`.
**Config chain value:** `qwen3.json` preset has `"rmsNormWeightOffset": false`.
**Contradiction:** Yes — JS says `true`, preset says `false`.
**Impact:** Applies Gemma-style `(1 + w) * x` RMS norm instead of standard
`w * x`, causing ~2x scaling error at every norm layer.
**Fix:** Remove the guard. Let the preset drive the value.

### 2. `detectAttentionOutputGate` — MEDIUM

**File:** `src/converter/manifest-inference.js:235-248`

```javascript
function detectAttentionOutputGate(presetInference, modelConfig, defaults) {
  if (typeof presetInference?.attention?.attentionOutputGate === 'boolean') {
    return presetInference.attention.attentionOutputGate;
  }
  if (typeof modelConfig?.attn_output_gate === 'boolean') {
    return modelConfig.attn_output_gate;
  }
  if (isQwen35LinearAttentionConfig(modelConfig)) {
    return true;  // ← hardcoded, after preset but before defaults
  }
  return defaults.attention.attentionOutputGate;
}
```

**Gate:** `isQwen35LinearAttentionConfig` (model-type identity check).
**Hardcoded value:** `true`.
**Config chain value:** `qwen3.json` does not specify `attentionOutputGate`.
**Contradiction:** No (preset is silent), but the value should be in the preset.
**Fix:** Add `"attentionOutputGate": true` to `qwen3.json`, remove the JS guard.

## Output Format

```json
{
  "findings": [
    {
      "id": "CBA-001",
      "function": "detectRmsNormWeightOffset",
      "file": "src/converter/manifest-inference.js",
      "lines": [275, 278],
      "severity": "CRITICAL",
      "gate": "isQwen35LinearAttentionConfig (model-type identity)",
      "hardcodedValue": true,
      "configField": "normalization.rmsNormWeightOffset",
      "configSource": "src/config/presets/models/qwen3.json",
      "configValue": false,
      "contradiction": true,
      "firesBeforeConfigCheck": true,
      "fix": "Remove guard; preset already has correct value"
    }
  ],
  "filesScanned": [],
  "totalFindings": 0,
  "bySeverity": { "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0 }
}
```

## Scope

- Primary: `src/converter/manifest-inference.js` (highest density of detect/resolve functions)
- Secondary: `src/converter/*.js`, `src/inference/**/*.js`, `src/config/**/*.js`, `src/loader/**/*.js`
- Exclude: `tests/`, `tools/`, `benchmarks/`, `docs/`, `demo/`
