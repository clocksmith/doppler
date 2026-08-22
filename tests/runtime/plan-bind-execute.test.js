import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { hashTargetPlan } from '../../src/config/target-plan.js';
import {
  TEST_PACK_AUTHORITY,
  TEST_PACK_PUBLIC_KEY,
  createSignedPackFixture,
} from '../helpers/pack-v2-fixture.js';

const fixture = await createSignedPackFixture();
const buffers = [];
const writes = [];
const events = [];
const gpuDevice = {
  limits: { maxBufferSize: 1024 },
  createBuffer(descriptor) {
    const buffer = { descriptor, destroyed: false, destroy() { this.destroyed = true; } };
    buffers.push(buffer);
    return buffer;
  },
  createCommandEncoder() {},
  queue: {
    writeBuffer(buffer, offset, source, sourceOffset, size) {
      writes.push({ buffer, offset, source, sourceOffset, size });
    },
  },
};
const program = {
  executionGraphHash: fixture.pack.program.executionGraphHash,
  tokenize() { return [1, 2, 3]; },
  decodeTokens(tokens) { return tokens.join(','); },
  getTokenContract() { return { padTokenId: null, eosTokenId: null, stopTokenIds: [] }; },
  reset() {},
  async executePhase(phase, request) {
    const next = phase === 'prefill' ? 4 : request.context.contextTokens.at(-1) + 1;
    const logits = new Float32Array(8).fill(-10);
    logits[next] = 10;
    return { logits };
  },
  releaseStepResult() {},
  async close() {},
};
const runtime = createDopplerRuntime({
  device: {
    getDevice: () => gpuDevice,
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }),
  },
  artifactStore: fixture.artifactStore,
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  observer: { observe(event) { events.push(event.type); } },
  async programFactory() { return program; },
});
const session = await runtime.openPack(fixture.pack);
const before = hashTargetPlan(session.selectedPlan);
const tokens = [];
for await (const token of session.generate({
  promptTokens: [1, 2, 3], maxTokens: 4, maxSeqLen: 16,
  temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1,
  repetitionPenaltyWindow: 8, seed: 0, useChatTemplate: false,
})) tokens.push(token);
assert.deepEqual(tokens, [4, 5, 6, 7]);
assert.equal(buffers.length, 1, 'ResourceBinder must allocate a physical GPU buffer');
assert.equal(writes.length, 1, 'ResourceBinder must upload prompt token IDs');
assert.equal(hashTargetPlan(session.selectedPlan), before);
assert.deepEqual(events.slice(0, 3), ['pack-validation-started', 'pack-validation-complete', 'target-selected']);
await session.close();
assert.equal(buffers[0].destroyed, true);
assert.equal(hashTargetPlan(session.selectedPlan), before);

console.log('✔ plan-bind-execute.test.js passed');
