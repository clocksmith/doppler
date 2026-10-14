import assert from 'node:assert/strict';
import { mergeConfig } from '../../src/config/merge.js';
import {
  DEFAULT_MANIFEST_INFERENCE,
  resolveLinearNormGateActivation,
} from '../../src/config/schema/manifest.schema.js';
import { validateCallTimeOptions } from '../../src/config/param-validator.js';
import { validateRequiredInferenceFields } from '../../src/config/manifest-inference-validation.js';
import { parseModelConfig } from '../../src/inference/pipelines/text/config.js';
import { createExecutionV1Session } from '../helpers/execution-v1-fixtures.js';

// Contract: inference.normalization.linearNormGateActivation is source-owned.
// An authored value reaches ParsedModelConfig unchanged; a pre-field manifest
// resolves to the schema default silu; runtime/call-time overrides and unknown
// values fail instead of being rewritten.
function manifestWith(normalizationPatch) {
  const inference = structuredClone(DEFAULT_MANIFEST_INFERENCE);
  delete inference.normalization.linearNormGateActivation;
  Object.assign(inference.normalization, normalizationPatch);
  inference.session = createExecutionV1Session();
  return {
    version: 1,
    modelId: 'linear-norm-gate-contract',
    modelType: 'transformer',
    quantization: 'F16',
    architecture: {
      numLayers: 1, hiddenSize: 64, intermediateSize: 128, numAttentionHeads: 4, numKeyValueHeads: 4,
      headDim: 16, vocabSize: 32, maxSeqLen: 64,
      linearNumKeyHeads: 2, linearNumValueHeads: 2, linearKeyHeadDim: 16, linearValueHeadDim: 16,
      linearConvKernelDim: 4, linearNormMode: 'shared',
    },
    inference,
    eos_token_id: 1,
    tensors: {},
  };
}

// Pre-field manifest: the only legacy meaning is the schema default.
assert.equal(DEFAULT_MANIFEST_INFERENCE.normalization.linearNormGateActivation, 'silu');
{
  const legacy = manifestWith({});
  assert.equal(legacy.inference.normalization.linearNormGateActivation, undefined);
  const merged = mergeConfig(legacy);
  assert.equal(merged.inference.normalization.linearNormGateActivation, 'silu');
  assert.equal(merged._sources.get('inference.normalization.linearNormGateActivation'), 'manifest');
  assert.equal(parseModelConfig(legacy, {}).linearNormGateActivation, 'silu');
  assert.equal(legacy.inference.normalization.linearNormGateActivation, undefined, 'merge must not write back into the manifest');
}

// Authored values survive manifest -> merge -> ParsedModelConfig without rewrite.
for (const activation of ['sigmoid', 'silu']) {
  const manifest = manifestWith({ linearNormGateActivation: activation });
  const merged = mergeConfig(manifest);
  assert.equal(merged.inference.normalization.linearNormGateActivation, activation);
  assert.equal(merged._sources.get('inference.normalization.linearNormGateActivation'), 'manifest');
  const parsed = parseModelConfig(manifest, {});
  assert.equal(parsed.linearNormGateActivation, activation);
  assert.equal(parsed.outputGateType, null, 'the norm gate is never mirrored into the full-attention gate field');
}

// Source-owned: runtime overrides fail rather than rewrite the manifest.
for (const override of ['silu', 'sigmoid', null]) {
  assert.throws(
    () => mergeConfig(manifestWith({ linearNormGateActivation: 'sigmoid' }), { normalization: { linearNormGateActivation: override } }),
    /linearNormGateActivation is source-owned/
  );
}
assert.throws(
  () => parseModelConfig(manifestWith({}), { normalization: { linearNormGateActivation: 'sigmoid' } }),
  /modelOverrides may only override|source-owned/
);
assert.throws(() => validateCallTimeOptions({ linearNormGateActivation: 'sigmoid' }), /is a model param/);

// Malformed values fail; null is not a request for the legacy default.
for (const invalid of [null, 'swish', 'SIGMOID', 1]) {
  assert.throws(() => mergeConfig(manifestWith({ linearNormGateActivation: invalid })), /must be one of silu, sigmoid/);
  assert.throws(() => resolveLinearNormGateActivation({ linearNormGateActivation: invalid }), /must be one of/);
  const inference = manifestWith({ linearNormGateActivation: invalid }).inference;
  assert.throws(() => validateRequiredInferenceFields(inference, 'invalid-gate'), /linearNormGateActivation must be one of/);
}
assert.throws(() => resolveLinearNormGateActivation(undefined), /must be an object/);
assert.doesNotThrow(() => validateRequiredInferenceFields(manifestWith({}).inference, 'legacy-gate'));

console.log('linear-norm-gate-activation.test: ok');
