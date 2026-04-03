import assert from 'node:assert/strict';

const {
  mergeConfig,
  formatConfigSources,
  getValuesBySource,
  summarizeSources,
} = await import('../../src/config/merge.js');

// Minimal valid manifest for mergeConfig
function createManifest(overrides = {}) {
  return {
    modelId: 'test-model',
    architecture: 'transformer',
    inference: {
      layerPattern: 'standard',
      chatTemplate: { type: 'none', enabled: false },
      pipeline: 'text',
      attention: {
        queryPreAttnScalar: 1.0,
        attentionBias: false,
        attnLogitSoftcapping: null,
        slidingWindow: null,
        queryKeyNorm: false,
        attentionOutputGate: false,
        causal: true,
      },
      normalization: {
        rmsNormEps: 1e-6,
        rmsNormWeightOffset: false,
        postAttentionNorm: false,
        preFeedforwardNorm: false,
        postFeedforwardNorm: false,
      },
      ffn: {
        activation: 'gelu',
        gatedActivation: true,
        swigluLimit: null,
      },
      rope: {
        ropeTheta: 10000,
        ropeLocalTheta: null,
        mropeInterleaved: false,
        mropeSection: null,
        partialRotaryFactor: 1.0,
        ropeLocalPartialRotaryFactor: null,
        ropeFrequencyBaseDim: null,
        ropeLocalFrequencyBaseDim: null,
        ropeScalingType: null,
        ropeScalingFactor: null,
        ropeLocalScalingType: null,
        ropeLocalScalingFactor: null,
        yarnBetaFast: null,
        yarnBetaSlow: null,
        yarnOriginalMaxPos: null,
        ropeLocalYarnBetaFast: null,
        ropeLocalYarnBetaSlow: null,
        ropeLocalYarnOriginalMaxPos: null,
      },
      output: {
        finalLogitSoftcapping: null,
        tieWordEmbeddings: false,
        scaleEmbeddings: null,
        embeddingTranspose: false,
        embeddingVocabSize: null,
        embeddingPostprocessor: null,
      },
      ...overrides,
    },
  };
}

// === mergeConfig: manifest-only (no runtime overrides) ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, undefined);

  assert.equal(merged.modelId, 'test-model');
  assert.equal(merged.architecture, 'transformer');
  assert.equal(merged.inference.layerPattern, 'standard');
  assert.equal(merged.inference.attention.causal, true);
  assert.equal(merged.inference.normalization.rmsNormEps, 1e-6);
  assert.equal(merged.inference.ffn.activation, 'gelu');
  assert.equal(merged.inference.rope.ropeTheta, 10000);
  assert.equal(merged.inference.output.tieWordEmbeddings, false);
  assert.equal(merged.inference.chatTemplate.type, 'none');
  assert.ok(merged._sources instanceof Map);

  // All sources should be 'manifest' when no runtime overrides
  for (const source of merged._sources.values()) {
    assert.equal(source, 'manifest');
  }
}

// === mergeConfig: runtime override replaces manifest values ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    layerPattern: 'custom',
    pipeline: 'embedding',
    attention: { causal: false, slidingWindow: 512 },
    normalization: { rmsNormEps: 1e-5 },
    ffn: { activation: 'silu' },
    rope: { ropeTheta: 500000 },
    output: {
      tieWordEmbeddings: true,
      embeddingPostprocessor: {
        poolingMode: 'mean',
        includePrompt: true,
        projections: [],
        normalize: 'l2',
      },
    },
    chatTemplate: { enabled: true },
  });

  assert.equal(merged.inference.layerPattern, 'custom');
  assert.equal(merged.inference.pipeline, 'embedding');
  assert.equal(merged.inference.attention.causal, false);
  assert.equal(merged.inference.attention.slidingWindow, 512);
  assert.equal(merged.inference.normalization.rmsNormEps, 1e-5);
  assert.equal(merged.inference.ffn.activation, 'silu');
  assert.equal(merged.inference.rope.ropeTheta, 500000);
  assert.equal(merged.inference.output.tieWordEmbeddings, true);
  assert.equal(merged.inference.output.embeddingPostprocessor?.poolingMode, 'mean');
  assert.equal(merged.inference.chatTemplate.enabled, true);

  // Overridden fields should be tracked as 'runtime'
  assert.equal(merged._sources.get('inference.layerPattern'), 'runtime');
  assert.equal(merged._sources.get('inference.pipeline'), 'runtime');
  assert.equal(merged._sources.get('inference.attention.causal'), 'runtime');
  assert.equal(merged._sources.get('inference.normalization.rmsNormEps'), 'runtime');
  assert.equal(merged._sources.get('inference.rope.ropeTheta'), 'runtime');
  assert.equal(merged._sources.get('inference.output.embeddingPostprocessor'), 'runtime');

  // Non-overridden fields should remain 'manifest'
  assert.equal(merged._sources.get('inference.attention.queryPreAttnScalar'), 'manifest');
  assert.equal(merged._sources.get('inference.ffn.gatedActivation'), 'manifest');
}

// === mergeConfig: runtime with undefined values keeps manifest ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    attention: { causal: undefined },
  });

  // undefined runtime value should not override manifest
  assert.equal(merged.inference.attention.causal, true);
  assert.equal(merged._sources.get('inference.attention.causal'), 'manifest');
}

// === mergeConfig: runtime with falsy values (0, false, null, '') ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    attention: { queryPreAttnScalar: 0 },
    normalization: { postAttentionNorm: false },
    output: { finalLogitSoftcapping: null },
  });

  // 0, false, null should all override (they are defined)
  assert.equal(merged.inference.attention.queryPreAttnScalar, 0);
  assert.equal(merged._sources.get('inference.attention.queryPreAttnScalar'), 'runtime');
  assert.equal(merged.inference.normalization.postAttentionNorm, false);
  assert.equal(merged._sources.get('inference.normalization.postAttentionNorm'), 'runtime');
  assert.equal(merged.inference.output.finalLogitSoftcapping, null);
  assert.equal(merged._sources.get('inference.output.finalLogitSoftcapping'), 'runtime');
}

// === mergeConfig: all RoPE fields merge ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    rope: {
      ropeTheta: 1000000,
      ropeLocalTheta: 50000,
      ropeFrequencyBaseDim: 256,
      yarnBetaFast: 32,
      yarnBetaSlow: 1,
      yarnOriginalMaxPos: 8192,
    },
  });

  assert.equal(merged.inference.rope.ropeTheta, 1000000);
  assert.equal(merged.inference.rope.ropeLocalTheta, 50000);
  assert.equal(merged.inference.rope.ropeFrequencyBaseDim, 256);
  assert.equal(merged.inference.rope.yarnBetaFast, 32);
  assert.equal(merged.inference.rope.yarnBetaSlow, 1);
  assert.equal(merged.inference.rope.yarnOriginalMaxPos, 8192);
  // Non-overridden RoPE fields remain from manifest
  assert.equal(merged.inference.rope.partialRotaryFactor, 1.0);
  assert.equal(merged._sources.get('inference.rope.partialRotaryFactor'), 'manifest');
}

// === formatConfigSources ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    attention: { causal: false },
  });

  const formatted = formatConfigSources(merged);
  assert.equal(typeof formatted, 'string');
  assert.ok(formatted.includes('inference.attention.causal: false (runtime)'));
  assert.ok(formatted.includes('inference.attention.queryPreAttnScalar: 1 (manifest)'));

  // Output should be sorted
  const lines = formatted.split('\n');
  const sortedLines = [...lines].sort();
  assert.deepEqual(lines, sortedLines);
}

// === getValuesBySource ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    attention: { causal: false },
    rope: { ropeTheta: 999 },
  });

  const runtimeValues = getValuesBySource(merged, 'runtime');
  assert.ok(Array.isArray(runtimeValues));
  assert.ok(runtimeValues.length >= 2);

  const runtimePaths = runtimeValues.map(([path]) => path);
  assert.ok(runtimePaths.includes('inference.attention.causal'));
  assert.ok(runtimePaths.includes('inference.rope.ropeTheta'));

  // Verify values match
  const causalEntry = runtimeValues.find(([p]) => p === 'inference.attention.causal');
  assert.equal(causalEntry[1], false);

  const manifestValues = getValuesBySource(merged, 'manifest');
  assert.ok(manifestValues.length > 0);
  const manifestPaths = manifestValues.map(([path]) => path);
  assert.ok(manifestPaths.includes('inference.attention.queryPreAttnScalar'));
}

// === summarizeSources ===

{
  const manifest = createManifest();
  const merged = mergeConfig(manifest, {
    attention: { causal: false },
  });

  const summary = summarizeSources(merged);
  assert.equal(typeof summary.manifest, 'number');
  assert.equal(typeof summary.runtime, 'number');
  assert.ok(summary.manifest > 0);
  assert.ok(summary.runtime >= 1);
  assert.equal(summary.manifest + summary.runtime, merged._sources.size);
}

{
  // No overrides means all manifest
  const manifest = createManifest();
  const merged = mergeConfig(manifest, undefined);
  const summary = summarizeSources(merged);
  assert.equal(summary.runtime, 0);
  assert.ok(summary.manifest > 0);
}

console.log('config-merge.test: ok');
