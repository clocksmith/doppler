import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const { mergeConfig, summarizeSources } = await import('../../src/config/merge.js');
const { resolvePreset, PRESET_REGISTRY } = await import('../../src/config/loader.js');
const { DEFAULT_PRESET_INFERENCE_CONFIG } = await import('../../src/config/schema/index.js');

// =============================================================================
// Layer 1 — Stable merge contract sweep
//
// One explicit witness manifest with every overrideable field present.
// For each field: manifest-only, runtime-override, and runtime-undefined.
// Static OVERRIDE_FIELDS list must match the actual merge surface.
// =============================================================================

// Every source-tracked path that mergeConfig produces, grouped by domain.
// If someone adds a field to merge.js, this list must be updated or the
// meta-guard at the bottom fails.
const OVERRIDE_FIELDS = [
  // top-level
  { path: 'inference.layerPattern', domain: null, field: 'layerPattern', witness: 'standard', sentinel: 'custom' },
  { path: 'inference.defaultKernelPath', domain: null, field: 'defaultKernelPath', witness: 'witness-kp', sentinel: 'override-kp' },
  { path: 'inference.pipeline', domain: null, field: 'pipeline', witness: 'text', sentinel: 'embedding' },
  // attention
  { path: 'inference.attention.queryPreAttnScalar', domain: 'attention', field: 'queryPreAttnScalar', witness: 1.0, sentinel: 256 },
  { path: 'inference.attention.attentionBias', domain: 'attention', field: 'attentionBias', witness: false, sentinel: true },
  { path: 'inference.attention.attnLogitSoftcapping', domain: 'attention', field: 'attnLogitSoftcapping', witness: null, sentinel: 50.0 },
  { path: 'inference.attention.slidingWindow', domain: 'attention', field: 'slidingWindow', witness: null, sentinel: 4096 },
  { path: 'inference.attention.queryKeyNorm', domain: 'attention', field: 'queryKeyNorm', witness: false, sentinel: true },
  { path: 'inference.attention.attentionOutputGate', domain: 'attention', field: 'attentionOutputGate', witness: false, sentinel: true },
  { path: 'inference.attention.causal', domain: 'attention', field: 'causal', witness: true, sentinel: false },
  // normalization
  { path: 'inference.normalization.rmsNormEps', domain: 'normalization', field: 'rmsNormEps', witness: 1e-6, sentinel: 1e-5 },
  { path: 'inference.normalization.rmsNormWeightOffset', domain: 'normalization', field: 'rmsNormWeightOffset', witness: false, sentinel: true },
  { path: 'inference.normalization.postAttentionNorm', domain: 'normalization', field: 'postAttentionNorm', witness: false, sentinel: true },
  { path: 'inference.normalization.preFeedforwardNorm', domain: 'normalization', field: 'preFeedforwardNorm', witness: false, sentinel: true },
  { path: 'inference.normalization.postFeedforwardNorm', domain: 'normalization', field: 'postFeedforwardNorm', witness: false, sentinel: true },
  // ffn
  { path: 'inference.ffn.activation', domain: 'ffn', field: 'activation', witness: 'gelu', sentinel: 'silu' },
  { path: 'inference.ffn.gatedActivation', domain: 'ffn', field: 'gatedActivation', witness: true, sentinel: false },
  { path: 'inference.ffn.swigluLimit', domain: 'ffn', field: 'swigluLimit', witness: null, sentinel: 8192 },
  // rope
  { path: 'inference.rope.ropeTheta', domain: 'rope', field: 'ropeTheta', witness: 10000, sentinel: 1000000 },
  { path: 'inference.rope.ropeLocalTheta', domain: 'rope', field: 'ropeLocalTheta', witness: null, sentinel: 50000 },
  { path: 'inference.rope.mropeInterleaved', domain: 'rope', field: 'mropeInterleaved', witness: false, sentinel: true },
  { path: 'inference.rope.mropeSection', domain: 'rope', field: 'mropeSection', witness: null, sentinel: [16, 24, 24] },
  { path: 'inference.rope.partialRotaryFactor', domain: 'rope', field: 'partialRotaryFactor', witness: 1.0, sentinel: 0.25 },
  { path: 'inference.rope.ropeScalingType', domain: 'rope', field: 'ropeScalingType', witness: null, sentinel: 'yarn' },
  { path: 'inference.rope.ropeScalingFactor', domain: 'rope', field: 'ropeScalingFactor', witness: null, sentinel: 4.0 },
  { path: 'inference.rope.ropeLocalScalingType', domain: 'rope', field: 'ropeLocalScalingType', witness: null, sentinel: 'linear' },
  { path: 'inference.rope.ropeLocalScalingFactor', domain: 'rope', field: 'ropeLocalScalingFactor', witness: null, sentinel: 2.0 },
  { path: 'inference.rope.yarnBetaFast', domain: 'rope', field: 'yarnBetaFast', witness: null, sentinel: 32 },
  { path: 'inference.rope.yarnBetaSlow', domain: 'rope', field: 'yarnBetaSlow', witness: null, sentinel: 1 },
  { path: 'inference.rope.yarnOriginalMaxPos', domain: 'rope', field: 'yarnOriginalMaxPos', witness: null, sentinel: 8192 },
  { path: 'inference.rope.ropeLocalYarnBetaFast', domain: 'rope', field: 'ropeLocalYarnBetaFast', witness: null, sentinel: 16 },
  { path: 'inference.rope.ropeLocalYarnBetaSlow', domain: 'rope', field: 'ropeLocalYarnBetaSlow', witness: null, sentinel: 2 },
  { path: 'inference.rope.ropeLocalYarnOriginalMaxPos', domain: 'rope', field: 'ropeLocalYarnOriginalMaxPos', witness: null, sentinel: 4096 },
  // output
  { path: 'inference.output.finalLogitSoftcapping', domain: 'output', field: 'finalLogitSoftcapping', witness: null, sentinel: 30.0 },
  { path: 'inference.output.tieWordEmbeddings', domain: 'output', field: 'tieWordEmbeddings', witness: false, sentinel: true },
  { path: 'inference.output.scaleEmbeddings', domain: 'output', field: 'scaleEmbeddings', witness: null, sentinel: true },
  { path: 'inference.output.embeddingTranspose', domain: 'output', field: 'embeddingTranspose', witness: false, sentinel: true },
  { path: 'inference.output.embeddingVocabSize', domain: 'output', field: 'embeddingVocabSize', witness: null, sentinel: 256000 },
  // chatTemplate
  { path: 'inference.chatTemplate.type', domain: 'chatTemplate', field: 'type', witness: 'none', sentinel: 'gemma' },
  { path: 'inference.chatTemplate.enabled', domain: 'chatTemplate', field: 'enabled', witness: false, sentinel: true },
];

function witnessManifest() {
  const inf = {};
  for (const entry of OVERRIDE_FIELDS) {
    if (!entry.domain) {
      inf[entry.field] = entry.witness;
    } else {
      if (!inf[entry.domain]) inf[entry.domain] = {};
      inf[entry.domain][entry.field] = entry.witness;
    }
  }
  return { modelId: 'witness', architecture: 'transformer', inference: inf };
}

function resolve(obj, dotPath) {
  let v = obj;
  for (const p of dotPath.split('.')) v = v?.[p];
  return v;
}

// --- 1a. Manifest-only: all sources are 'manifest' ---
{
  const merged = mergeConfig(witnessManifest(), undefined);
  for (const entry of OVERRIDE_FIELDS) {
    const actual = resolve(merged, entry.path);
    assert.deepStrictEqual(actual, entry.witness, `manifest-only ${entry.path}`);
    assert.equal(merged._sources.get(entry.path), 'manifest', `manifest-only source ${entry.path}`);
  }
  const summary = summarizeSources(merged);
  assert.equal(summary.manifest, OVERRIDE_FIELDS.length);
  assert.equal(summary.runtime, 0);
}
console.log(`  ok: manifest-only (${OVERRIDE_FIELDS.length} fields)`);

// --- 1b. Runtime override: each field individually ---
for (const entry of OVERRIDE_FIELDS) {
  const override = entry.domain
    ? { [entry.domain]: { [entry.field]: entry.sentinel } }
    : { [entry.field]: entry.sentinel };
  const merged = mergeConfig(witnessManifest(), override);
  const actual = resolve(merged, entry.path);
  assert.deepStrictEqual(actual, entry.sentinel, `override ${entry.path}`);
  assert.equal(merged._sources.get(entry.path), 'runtime', `override source ${entry.path}`);
}
console.log(`  ok: per-field runtime override (${OVERRIDE_FIELDS.length} fields)`);

// --- 1c. Runtime undefined: manifest preserved ---
{
  const allUndefined = {};
  for (const entry of OVERRIDE_FIELDS) {
    if (entry.domain) {
      if (!allUndefined[entry.domain]) allUndefined[entry.domain] = {};
      allUndefined[entry.domain][entry.field] = undefined;
    } else {
      allUndefined[entry.field] = undefined;
    }
  }
  const merged = mergeConfig(witnessManifest(), allUndefined);
  for (const entry of OVERRIDE_FIELDS) {
    assert.deepStrictEqual(resolve(merged, entry.path), entry.witness, `undefined-passthrough ${entry.path}`);
    assert.equal(merged._sources.get(entry.path), 'manifest', `undefined source ${entry.path}`);
  }
}
console.log('  ok: undefined passthrough');

// --- 1d. Falsy overrides (false, 0, null) ---
{
  const merged = mergeConfig(witnessManifest(), {
    attention: { queryPreAttnScalar: 0, causal: false, slidingWindow: null },
    rope: { ropeTheta: 0 },
    output: { scaleEmbeddings: false },
  });
  assert.equal(resolve(merged, 'inference.attention.queryPreAttnScalar'), 0);
  assert.equal(resolve(merged, 'inference.attention.causal'), false);
  assert.equal(resolve(merged, 'inference.attention.slidingWindow'), null);
  assert.equal(resolve(merged, 'inference.rope.ropeTheta'), 0);
  assert.equal(resolve(merged, 'inference.output.scaleEmbeddings'), false);
  for (const p of [
    'inference.attention.queryPreAttnScalar',
    'inference.attention.causal',
    'inference.attention.slidingWindow',
    'inference.rope.ropeTheta',
    'inference.output.scaleEmbeddings',
  ]) {
    assert.equal(merged._sources.get(p), 'runtime', `falsy source ${p}`);
  }
}
console.log('  ok: falsy overrides (0, false, null)');

// --- 1e. All overridden at once ---
{
  const bulk = {};
  for (const entry of OVERRIDE_FIELDS) {
    if (entry.domain) {
      if (!bulk[entry.domain]) bulk[entry.domain] = {};
      bulk[entry.domain][entry.field] = entry.sentinel;
    } else {
      bulk[entry.field] = entry.sentinel;
    }
  }
  const merged = mergeConfig(witnessManifest(), bulk);
  const summary = summarizeSources(merged);
  assert.equal(summary.runtime, OVERRIDE_FIELDS.length);
  assert.equal(summary.manifest, 0);
}
console.log('  ok: all fields overridden at once');

// --- 1f. _sources contains exactly the expected paths ---
{
  const merged = mergeConfig(witnessManifest(), undefined);
  const expectedPaths = new Set(OVERRIDE_FIELDS.map((e) => e.path));
  const actualPaths = new Set(merged._sources.keys());
  for (const p of expectedPaths) {
    assert.ok(actualPaths.has(p), `expected path missing from _sources: ${p}`);
  }
  for (const p of actualPaths) {
    assert.ok(expectedPaths.has(p), `unexpected path in _sources: ${p}`);
  }
}
console.log('  ok: _sources paths match OVERRIDE_FIELDS exactly');

// =============================================================================
// Meta-guard: OVERRIDE_FIELDS must match the actual merge surface.
//
// Parse merge.js for all overlay() and sources.set() calls, extract paths,
// and assert the test list is complete.
// =============================================================================
{
  const mergeSource = readFileSync(
    new URL('../../src/config/merge.js', import.meta.url),
    'utf8'
  );

  // Collect all source-tracked paths from merge.js.
  // Two patterns:
  //   1. sources.set('inference.foo.bar', ...) — top-level fields
  //   2. overlay(`${prefix}.fieldName`, ...) inside a function with
  //      const prefix = 'inference.domain' — domain fields
  const foundPaths = new Set();

  // Pattern 1: literal sources.set paths
  const sourcesSetPattern = /sources\.set\(\s*'([^']+)'/g;
  let match;
  while ((match = sourcesSetPattern.exec(mergeSource)) !== null) {
    foundPaths.add(match[1]);
  }

  // Pattern 2: overlay calls with template literals — resolve prefix from context.
  // Each merge function declares: const prefix = 'inference.domain';
  // Then calls: overlay(`${prefix}.fieldName`, ...)
  const prefixBlocks = mergeSource.matchAll(
    /const prefix = '([^']+)';\s*\n\s*return \{([\s\S]*?)\n\s*\};/g
  );
  for (const block of prefixBlocks) {
    const prefix = block[1];
    const body = block[2];
    const fieldPattern = /overlay\(\s*`\$\{prefix\}\.(\w+)`/g;
    let fieldMatch;
    while ((fieldMatch = fieldPattern.exec(body)) !== null) {
      foundPaths.add(`${prefix}.${fieldMatch[1]}`);
    }
  }

  const expectedPaths = new Set(OVERRIDE_FIELDS.map((e) => e.path));
  for (const p of foundPaths) {
    assert.ok(
      expectedPaths.has(p),
      `merge.js has path "${p}" not covered in OVERRIDE_FIELDS — add it to the test`
    );
  }
  for (const p of expectedPaths) {
    assert.ok(
      foundPaths.has(p),
      `OVERRIDE_FIELDS has path "${p}" not found in merge.js — stale entry`
    );
  }
}
console.log('  ok: meta-guard — OVERRIDE_FIELDS matches merge.js surface');

// =============================================================================
// Layer 2 — Family population sweep
//
// Resolve real presets, build minimal manifests, assert that family-specific
// fields are populated with non-undefined values.
// =============================================================================

function buildManifestFromPreset(presetId) {
  const preset = resolvePreset(presetId);
  const inf = preset.inference ?? {};
  const defaults = DEFAULT_PRESET_INFERENCE_CONFIG;

  // Build a manifest by layering preset inference over defaults,
  // matching what the converter does.
  function layer(base, over) {
    if (!over) return { ...base };
    const out = { ...base };
    for (const [k, v] of Object.entries(over)) {
      if (v !== undefined) out[k] = v;
    }
    return out;
  }

  return {
    modelId: `${presetId}-test`,
    architecture: preset.architecture ?? 'transformer',
    inference: {
      attention: layer(defaults.attention, inf.attention),
      normalization: layer(defaults.normalization, inf.normalization),
      ffn: layer(defaults.ffn, inf.ffn),
      rope: layer(defaults.rope, inf.rope),
      output: layer(defaults.output, inf.output),
      chatTemplate: layer(defaults.chatTemplate ?? { type: null, enabled: false }, inf.chatTemplate),
      layerPattern: inf.layerPattern ?? defaults.layerPattern ?? null,
      defaultKernelPath: inf.defaultKernelPath ?? defaults.defaultKernelPath ?? null,
      pipeline: inf.pipeline ?? defaults.pipeline ?? null,
    },
  };
}

const FAMILY_CHECKS = [
  {
    presetId: 'gemma3',
    populated: [
      'inference.attention.queryPreAttnScalar',
      'inference.attention.slidingWindow',
      'inference.attention.queryKeyNorm',
      'inference.normalization.postAttentionNorm',
      'inference.normalization.postFeedforwardNorm',
      'inference.rope.ropeTheta',
      'inference.rope.ropeLocalTheta',
      'inference.output.scaleEmbeddings',
      'inference.chatTemplate.type',
    ],
  },
  {
    presetId: 'qwen3',
    populated: [
      'inference.attention.causal',
      'inference.rope.ropeTheta',
      'inference.chatTemplate.type',
    ],
  },
  {
    presetId: 'gemma2',
    populated: [
      'inference.attention.attnLogitSoftcapping',
      'inference.attention.slidingWindow',
      'inference.output.finalLogitSoftcapping',
      'inference.rope.ropeTheta',
    ],
  },
  {
    presetId: 'mixtral',
    populated: [
      'inference.rope.ropeTheta',
      'inference.ffn.activation',
    ],
  },
  {
    presetId: 'llama3',
    populated: [
      'inference.rope.ropeTheta',
      'inference.ffn.activation',
      'inference.chatTemplate.type',
    ],
  },
];

for (const { presetId, populated } of FAMILY_CHECKS) {
  const manifest = buildManifestFromPreset(presetId);
  const merged = mergeConfig(manifest, undefined);

  for (const fieldPath of populated) {
    const val = resolve(merged, fieldPath);
    assert.ok(
      val !== undefined,
      `${presetId}: ${fieldPath} should be populated, got undefined`
    );
  }

  // Verify a single override works on the real preset shape
  const override = { attention: { causal: !merged.inference.attention.causal } };
  const overridden = mergeConfig(manifest, override);
  assert.equal(
    overridden.inference.attention.causal,
    !merged.inference.attention.causal,
    `${presetId}: causal override failed`
  );
  assert.equal(
    overridden._sources.get('inference.attention.causal'),
    'runtime',
    `${presetId}: causal source after override`
  );
}
console.log(`  ok: family population sweep (${FAMILY_CHECKS.length} families)`);

console.log('model-override-sweep.test: ok');
