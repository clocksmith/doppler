import assert from 'node:assert/strict';

import { createModelHandle } from '../../src/client/runtime/model-session.js';
import {
  assertArtifactVariantAllowed,
  resolveResolutionPolicy,
} from '../../src/client/runtime/resolution-policy.js';

const artifactId = `sha256:${'a'.repeat(64)}`;
const alternateArtifactId = `sha256:${'c'.repeat(64)}`;

const normalized = resolveResolutionPolicy({
  allowedArtifactVariantIds: [artifactId.toUpperCase(), artifactId],
  allowedExecutionIds: null,
});
assert.deepEqual(normalized.allowedArtifactVariantIds, [artifactId]);
assert.equal(Object.isFrozen(normalized), true);
assert.equal(Object.isFrozen(normalized.allowedArtifactVariantIds), true);
assert.throws(
  () => resolveResolutionPolicy({ allowedExecutions: [artifactId] }),
  /Unknown Doppler resolutionPolicy field/
);
assert.throws(
  () => resolveResolutionPolicy({ allowedExecutionIds: ['not-a-digest'] }),
  /requires SHA-256 identities/
);
assert.doesNotThrow(() => assertArtifactVariantAllowed(normalized, artifactId));
assert.throws(
  () => assertArtifactVariantAllowed(normalized, alternateArtifactId),
  /rejected artifact variant/
);
assert.throws(
  () => assertArtifactVariantAllowed(
    resolveResolutionPolicy({ allowedArtifactVariantIds: [] }),
    artifactId
  ),
  /rejected artifact variant/
);

let executions = 0;
const pipeline = {
  modelConfig: { chatTemplateEnabled: false },
  manifest: { inference: { chatTemplate: { enabled: false } } },
  runtimeConfig: {
    inference: {
      generation: { maxTokens: 1, useSpeculative: false },
      sampling: {
        temperature: 0,
        topP: 1,
        topK: 1,
        repetitionPenalty: 1,
        repetitionPenaltyWindow: 0,
        greedyThreshold: 0,
        suppressSpecialTokens: true,
        suppressSpecialLikeTokens: true,
        suppressTokenIds: [],
      },
      chatTemplate: { enabled: false },
    },
  },
  resolvedRuntimeSession: { id: `sha256:${'b'.repeat(64)}` },
  isLoaded: true,
  tokenizer: {
    encode: () => [1],
    decode: () => 'verified',
  },
  async generateTokenIds() {
    executions += 1;
    return { tokenIds: [1], stats: null };
  },
  getKernelCapabilities() {
    return {
      adapterInfo: {},
      hasF16: false,
      hasSubgroups: false,
      maxBufferSize: 0,
      deviceEpoch: 0,
    };
  },
  generate() {
    throw new Error('raw generation should remain unreachable');
  },
  unload() {},
};

const source = {
  logicalModelId: 'logical-model',
  modelId: 'resolved-model',
  manifestHash: artifactId,
};
const baseline = await createModelHandle(pipeline, source).generateWithEvidence('prompt');
const executionId = baseline.resolution.resolvedExecutionId;

const pinned = createModelHandle(pipeline, {
  ...source,
  resolutionPolicy: {
    allowedArtifactVariantIds: [artifactId],
    allowedExecutionIds: [executionId],
  },
});
assert.deepEqual(pinned.resolutionPolicy.allowedExecutionIds, [executionId]);
assert.equal(
  (await pinned.generateWithEvidence('prompt')).resolution.resolvedExecutionId,
  executionId
);
assert.throws(
  () => pinned.generate('prompt'),
  /may expose output before the final execution identity is verified/
);
assert.throws(
  () => createModelHandle(pipeline, {
    ...source,
    resolutionPolicy: { allowedArtifactVariantIds: [alternateArtifactId] },
  }),
  /rejected artifact variant/
);

const rejected = createModelHandle(pipeline, {
  ...source,
  resolutionPolicy: { allowedExecutionIds: [`sha256:${'d'.repeat(64)}`] },
});
await assert.rejects(
  () => rejected.generateWithEvidence('prompt'),
  /rejected execution/
);

const rejectAll = createModelHandle(pipeline, {
  ...source,
  resolutionPolicy: { allowedExecutionIds: [] },
});
const executionsBeforeRejectAll = executions;
await assert.rejects(
  () => rejectAll.generateWithEvidence('prompt'),
  /rejects every execution/
);
assert.equal(executions, executionsBeforeRejectAll);
