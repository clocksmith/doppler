import assert from 'node:assert/strict';

import { buildReferenceTranscriptSeed } from '../../src/inference/browser-harness/report.js';

function createRun(overrides = {}) {
  return {
    prompt: 'The sky is',
    promptInput: 'The sky is',
    promptTokenIds: [1, 2, 3],
    output: ' blue',
    tokenIds: [4],
    tokenDiagnostics: { preview: [] },
    logitsDigests: [],
    generationConfig: {
      maxTokens: 1,
      temperature: 0,
      topP: 1,
      topK: 1,
      repetitionPenalty: 1,
      repetitionPenaltyWindow: null,
      greedyThreshold: null,
      suppressSpecialTokens: true,
      suppressSpecialLikeTokens: true,
      suppressTokenIds: [],
      seed: null,
      useChatTemplate: false,
    },
    phase: {
      prefillMs: 1,
      decodeMs: 2,
      prefillTokens: 3,
      decodeTokens: 1,
      stopReason: 'max_tokens',
      stopTokenId: null,
      kvCache: {
        layout: 'contiguous',
        kvDtype: 'f16',
        seqLen: 4,
        maxSeqLen: 16,
        usedBytes: 32,
        allocatedBytes: 128,
      },
    },
    ...overrides,
  };
}

const context = {
  executionGraphHash: `sha256:${'a'.repeat(64)}`,
  surface: 'browser-webgpu',
};
const first = buildReferenceTranscriptSeed(createRun(), context);
const repeated = buildReferenceTranscriptSeed(createRun({
  phase: {
    ...createRun().phase,
    prefillMs: 99,
    decodeMs: 101,
  },
}), context);

assert.notDeepEqual(first.phase, repeated.phase);
assert.equal(
  first.source.hash,
  repeated.source.hash,
  'observation timing must not change reference transcript source identity'
);

const changedOutput = buildReferenceTranscriptSeed(createRun({
  output: ' green',
  tokenIds: [5],
}), context);
assert.notEqual(first.source.hash, changedOutput.source.hash);

const changedGraph = buildReferenceTranscriptSeed(createRun(), {
  ...context,
  executionGraphHash: `sha256:${'b'.repeat(64)}`,
});
assert.notEqual(first.source.hash, changedGraph.source.hash);

console.log('browser-harness-reference-transcript.test: ok');
