import { describe, expect, it } from 'vitest';

import { parseModelConfig } from '../../src/inference/pipeline/config.js';
import {
  DEFAULT_MANIFEST_INFERENCE,
  type ManifestInferenceSchema,
  type ManifestSchema,
} from '../../src/config/schema/index.js';

function cloneInference(): ManifestInferenceSchema {
  return JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE)) as ManifestInferenceSchema;
}

function makeManifest(inference: ManifestInferenceSchema): ManifestSchema {
  return {
    version: 1,
    modelId: 'test-model',
    modelType: 'transformer',
    quantization: 'F16',
    shards: [],
    totalSize: 0,
    tensorsFile: 'tensors.json',
    tensorCount: 0,
    groups: {},
    architecture: {
      numLayers: 2,
      hiddenSize: 16,
      intermediateSize: 32,
      numAttentionHeads: 2,
      numKeyValueHeads: 2,
      headDim: 8,
      vocabSize: 128,
      maxSeqLen: 64,
      ropeTheta: 10000,
      rmsNormEps: 1e-5,
    },
    inference,
  };
}

describe('manifest inference validation', () => {
  it('throws when normalization flags are missing', () => {
    const inference = cloneInference();
    delete (inference.normalization as { postAttentionNorm?: boolean }).postAttentionNorm;
    delete (inference.normalization as { preFeedforwardNorm?: boolean }).preFeedforwardNorm;
    delete (inference.normalization as { postFeedforwardNorm?: boolean }).postFeedforwardNorm;

    const run = () => parseModelConfig(makeManifest(inference));
    expect(run).toThrow(/postAttentionNorm/);
    expect(run).toThrow(/preFeedforwardNorm/);
    expect(run).toThrow(/postFeedforwardNorm/);
  });

  it('treats null as explicit disable and undefined as missing', () => {
    const withNull = cloneInference();
    withNull.attention.slidingWindow = null;
    expect(() => parseModelConfig(makeManifest(withNull))).not.toThrow();

    const withUndefined = cloneInference();
    delete (withUndefined.attention as { slidingWindow?: number | null }).slidingWindow;
    expect(() => parseModelConfig(makeManifest(withUndefined))).toThrow(/slidingWindow/);
  });

  it('fails fast when alternating layerPattern lacks globalPattern', () => {
    const inference = cloneInference();
    inference.layerPattern = { type: 'alternating' };
    expect(() => parseModelConfig(makeManifest(inference))).toThrow(/globalPattern/);
  });

  it('propagates YARN params into parsed config', () => {
    const inference = cloneInference();
    inference.rope.ropeScalingType = 'yarn';
    inference.rope.ropeScalingFactor = 2.0;
    inference.rope.yarnBetaFast = 32;
    inference.rope.yarnBetaSlow = 1;
    inference.rope.yarnOriginalMaxPos = 4096;

    const parsed = parseModelConfig(makeManifest(inference));
    expect(parsed.ropeScaling).toEqual({
      type: 'yarn',
      factor: 2.0,
      beta_fast: 32,
      beta_slow: 1,
      original_max_position_embeddings: 4096,
    });
  });
});
