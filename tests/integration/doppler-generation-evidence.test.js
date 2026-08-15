import assert from 'node:assert/strict';

import { createModelHandle } from '../../src/client/runtime/model-session.js';

function canonicalize(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((entry) => canonicalize(entry)).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalize(value[key])}`
  )).join(',')}}`;
}

async function hashValue(value) {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(canonicalize(value))
  );
  const hex = Array.from(new Uint8Array(digest), (byte) => (
    byte.toString(16).padStart(2, '0')
  )).join('');
  return `sha256:${hex}`;
}

const runtimeConfig = {
  inference: {
    generation: {
      maxTokens: 32,
      useSpeculative: false,
    },
    sampling: {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      repetitionPenalty: 1.1,
      repetitionPenaltyWindow: 64,
      greedyThreshold: 0,
      suppressSpecialTokens: true,
      suppressSpecialLikeTokens: true,
      suppressTokenIds: [],
    },
    chatTemplate: {
      enabled: true,
    },
  },
};

const stats = {
  kernelPathId: 'qwen35-webgpu',
  kernelPathSource: 'execution-v1',
  executionPlan: {
    primary: {
      id: 'primary',
      kernelPathId: 'qwen35-webgpu',
      kernelPathSource: 'execution-v1',
      activationDtype: 'f16',
    },
    fallback: null,
    activePlanIdAtStart: 'primary',
    finalActivePlanId: 'primary',
    transitions: [],
  },
};

const pipeline = {
  modelConfig: {
    chatTemplateEnabled: true,
  },
  manifest: {
    inference: {
      chatTemplate: {
        enabled: true,
      },
    },
  },
  runtimeConfig,
  resolvedRuntimeSession: {
    id: `sha256:${'b'.repeat(64)}`,
  },
  isLoaded: true,
  tokenizer: {
    encode() {
      return [11, 12, 13];
    },
    decode(tokenIds) {
      return tokenIds.map((tokenId) => `[${tokenId}]`).join('');
    },
  },
  async generateTokenIds(prompt, options) {
    if (prompt === 'hello') {
      assert.deepEqual(options, {
        maxTokens: 4,
        temperature: 0,
        topK: 1,
        topP: 1,
        useChatTemplate: false,
      });
    } else {
      assert.deepEqual(prompt, [{ role: 'user', content: 'hello' }]);
      assert.deepEqual(options, { maxTokens: 2 });
    }
    return {
      tokenIds: [101, 202],
      stats,
    };
  },
  getKernelCapabilities() {
    return {
      adapterInfo: {
        vendor: 'test-vendor',
        architecture: 'test-architecture',
        device: 'test-device',
        description: 'test adapter',
      },
      hasF16: true,
      hasSubgroups: false,
      maxBufferSize: 4096,
      deviceEpoch: 3,
    };
  },
  generate() {},
  async embed(prompt) {
    assert.equal(prompt, 'embedding input');
    return {
      embedding: new Float32Array([0.25, 0.75]),
      tokens: [31, 32],
      seqLen: 2,
      embeddingMode: 'mean',
      phase: null,
    };
  },
  embedBatch() {},
  embedImage() {},
  embedAudio() {},
  transcribeImage() {},
  transcribeAudio() {},
  transcribeVideo() {},
  unload() {},
};

const handle = createModelHandle(pipeline, {
  logicalModelId: 'qwen-logical',
  modelId: 'qwen-test',
  manifestHash: 'a'.repeat(64),
});
const evidence = await handle.generateWithEvidence('hello', {
  maxTokens: 4,
  temperature: 0,
  topK: 1,
  topP: 1,
  useChatTemplate: false,
});

assert.equal(evidence.schema, 'doppler_generation_evidence/v1');
assert.equal(handle.logicalModelId, 'qwen-logical');
assert.equal(handle.resolvedArtifactVariantId, `sha256:${'a'.repeat(64)}`);
assert.equal(evidence.outputText, '[101][202]');
assert.deepEqual(evidence.tokenIds, [101, 202]);
assert.equal(evidence.generationConfig.maxTokens, 4);
assert.equal(evidence.generationConfig.temperature, 0);
assert.equal(evidence.generationConfig.topK, 1);
assert.equal(evidence.generationConfig.topP, 1);
assert.equal(evidence.generationConfig.repetitionPenalty, 1.1);
assert.equal(evidence.generationConfig.useChatTemplate, false);
assert.equal(
  evidence.generationConfigHash,
  await hashValue(evidence.generationConfig)
);
assert.equal(
  evidence.transcriptHash,
  await hashValue(evidence.transcript)
);
assert.equal(evidence.backendIdentity.backend, 'webgpu');
assert.equal(evidence.backendIdentity.kernelPathId, 'qwen35-webgpu');
assert.equal(evidence.backendIdentity.executionPlanId, 'primary');
assert.equal(evidence.backendIdentity.activationDtype, 'f16');
assert.equal(
  evidence.backendIdentityHash,
  await hashValue(evidence.backendIdentity)
);
assert.equal(
  evidence.runtimeProfileHash,
  await hashValue(evidence.runtimeProfile)
);
assert.equal(evidence.resolution.schema, 'doppler.resolution-identity/v1');
assert.equal(evidence.resolution.logicalModelId, 'qwen-logical');
assert.equal(evidence.resolution.resolvedArtifactVariantId, `sha256:${'a'.repeat(64)}`);
assert.equal(evidence.resolution.resolvedExecutionId, await hashValue(evidence.executionIdentity));
assert.equal(
  evidence.executionIdentity.resolvedRuntimeSessionId,
  `sha256:${'b'.repeat(64)}`
);
assert.equal(evidence.runtimeProfile.model.modelId, 'qwen-test');
assert.equal(evidence.runtimeProfile.model.manifestHash, `sha256:${'a'.repeat(64)}`);
assert.equal(
  evidence.runtimeProfile.resolvedRuntimeSessionId,
  `sha256:${'b'.repeat(64)}`
);

const chatResponse = await handle.chatText([
  { role: 'user', content: 'hello' },
], { maxTokens: 2 });
assert.equal(chatResponse.content, '[101][202]');
assert.deepEqual(chatResponse.usage, {
  promptTokens: 3,
  completionTokens: 2,
  totalTokens: 5,
});
assert.equal(chatResponse.evidence.resolution.logicalModelId, 'qwen-logical');

const embeddingEvidence = await handle.embedWithEvidence('embedding input');
assert.equal(embeddingEvidence.schema, 'doppler_embedding_evidence/v1');
assert.deepEqual(Array.from(embeddingEvidence.embedding), [0.25, 0.75]);
assert.equal(embeddingEvidence.resolution.logicalModelId, 'qwen-logical');
assert.equal(
  embeddingEvidence.resolution.resolvedExecutionId,
  await hashValue(embeddingEvidence.executionIdentity)
);
assert.equal(embeddingEvidence.executionIdentity.activeAdapter, null);
assert.equal(
  embeddingEvidence.inputHash,
  await hashValue({ text: 'embedding input' })
);
assert.equal(
  embeddingEvidence.outputHash,
  await hashValue({
    embedding: [0.25, 0.75],
    tokens: [31, 32],
    seqLen: 2,
    embeddingMode: 'mean',
  })
);

delete pipeline.runtimeConfig.inference.chatTemplate;
pipeline.generateTokenIds = async (_prompt, options) => ({
  tokenIds: [303],
  stats,
});
const modelDefaultEvidence = await handle.generateWithEvidence('hello', {
  maxTokens: 1,
});
assert.equal(modelDefaultEvidence.generationConfig.useChatTemplate, true);

delete pipeline.modelConfig.chatTemplateEnabled;
const disabledDefaultEvidence = await handle.generateWithEvidence('hello', {
  maxTokens: 1,
});
assert.equal(disabledDefaultEvidence.generationConfig.useChatTemplate, false);

const missingExecutionIdentityHandle = createModelHandle({
  ...pipeline,
  resolvedRuntimeSession: null,
}, {
  logicalModelId: 'qwen-logical',
  modelId: 'qwen-test',
  manifestHash: 'a'.repeat(64),
});
await assert.rejects(
  () => missingExecutionIdentityHandle.generateWithEvidence('hello', { maxTokens: 1 }),
  /requires resolvedRuntimeSessionId as a SHA-256 digest/
);

console.log('doppler-generation-evidence.test: ok');
