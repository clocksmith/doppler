import { describe, expect, it } from 'vitest';

import { mergeConfig } from '../../src/config/merge.js';
import {
  DEFAULT_MANIFEST_INFERENCE,
  type ManifestInferenceSchema,
} from '../../src/config/schema/index.js';

function cloneInference(): ManifestInferenceSchema {
  return JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE)) as ManifestInferenceSchema;
}

describe('mergeConfig', () => {
  it('prefers runtime overrides and tracks sources', () => {
    const inference = cloneInference();
    const merged = mergeConfig(
      { modelId: 'test-model', inference },
      {
        attention: { queryKeyNorm: true },
        output: { finalLogitSoftcapping: 42 },
        rope: { ropeTheta: 12345 },
      }
    );

    expect(merged.inference.attention.queryKeyNorm).toBe(true);
    expect(merged._sources.get('inference.attention.queryKeyNorm')).toBe('runtime');

    expect(merged.inference.output.finalLogitSoftcapping).toBe(42);
    expect(merged._sources.get('inference.output.finalLogitSoftcapping')).toBe('runtime');

    expect(merged.inference.rope.ropeTheta).toBe(12345);
    expect(merged._sources.get('inference.rope.ropeTheta')).toBe('runtime');

    expect(merged._sources.get('inference.output.tieWordEmbeddings')).toBe('manifest');
  });
});
