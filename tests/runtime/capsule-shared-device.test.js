import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import { createResourceBinder } from '../../src/client/runtime/resource-binder.js';
import { createSessionController } from '../../src/client/runtime/session-controller.js';
import { createSignedCapsuleFixture, TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY } from '../helpers/capsule-v2-fixture.js';

const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const fixture = await createSignedCapsuleFixture();
const options = { promptTokens: [1], maxTokens: 1, maxSeqLen: 16, temperature: 0, topP: 1, topK: 1,
  repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false, seed: 0 };
function deviceFixture() {
  const loss = deferred(), buffers = [];
  const device = { lost: loss.promise, buffers, failAllocation: false, destroyed: false,
    limits: { maxBufferSize: 1024 }, destroy() { this.destroyed = true; }, createCommandEncoder() {},
    createBuffer(descriptor) {
      if (this.failAllocation) throw new Error('injected allocation failure');
      const buffer = { ...descriptor, destroyed: false, destroy() { this.destroyed = true; } };
      buffers.push(buffer); return buffer;
    }, queue: { writeBuffer(buffer) { assert.equal(buffer.destroyed, false, 'cannot write a released buffer'); } } };
  return { device, loss };
}
function runtimeFixture(device, construction = async () => {}) {
  const programs = [];
  const runtime = createDopplerRuntime({ device: { getDevice: () => device,
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
    artifactStore: fixture.artifactStore, trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY },
    async programFactory({ options }) {
      const program = { executionGraphHash: fixture.capsule.program.executionGraphHash, gate: null, entered: null,
        owned: new Set(), closed: false, observations: [], tokenize: () => [1], decodeTokens: ids => ids.join(','),
        getTokenContract: () => ({ padTokenId: null, eosTokenId: null, stopTokenIds: [] }), reset() {},
        async executePhase(_phase, request) {
          this.observations.push(request.context.generationOptions.temperature);
          const buffer = device.createBuffer({ size: 32, usage: 1 }); this.owned.add(buffer);
          this.entered?.resolve(); await this.gate?.promise;
          const logits = new Float32Array(8).fill(-100); logits[4] = 100;
          return { logits, buffer };
        },
        releaseStepResult(result) { if (result) { result.buffer.destroy(); this.owned.delete(result.buffer); } },
        close() { this.closed = true; for (const buffer of this.owned) buffer.destroy(); this.owned.clear(); },
      };
      programs.push(program); await construction(options); return program;
    } });
  return { programs, open: options => runtime.openCapsule(fixture.capsule, options) };
}

test('closing a session waits for its pending readback and preserves another session on the same device', async () => {
  const { device } = deviceFixture(), runtime = runtimeFixture(device);
  const first = await runtime.open(), second = await runtime.open();
  const owner = runtime.programs[0]; owner.gate = deferred(); owner.entered = deferred();
  const pending = first.generateText(options); const rejected = assert.rejects(pending, /closed|aborted/);
  await owner.entered.promise;
  const live = [...device.buffers]; const closing = first.close();
  assert(live.every(buffer => !buffer.destroyed));
  assert.equal(owner.closed, false);
  const other = await second.generateText({ ...options, temperature: 0.5 });
  assert.deepEqual(other.tokenIds, [4]); assert.equal(device.destroyed, false);
  owner.gate.resolve(); await rejected; await closing;
  assert(live.every(buffer => buffer.destroyed)); assert.equal(runtime.programs[1].closed, false);
  assert.deepEqual((await second.generateText(options)).tokenIds, [4]);
  assert.deepEqual(runtime.programs[0].observations, [0]);
  assert.deepEqual(runtime.programs[1].observations, [0.5, 0]);
  await second.close(); assert.equal(device.destroyed, false);
});

test('device loss during readback prevents token delivery and invalidates both sessions', async () => {
  const { device, loss } = deviceFixture(), runtime = runtimeFixture(device);
  const first = await runtime.open(), second = await runtime.open();
  const owner = runtime.programs[0]; owner.gate = deferred(); owner.entered = deferred();
  const stream = first.generate(options), pending = stream.next();
  await owner.entered.promise; loss.resolve({ reason: 'unknown', message: 'injected loss' }); await device.lost;
  owner.gate.resolve();
  try {
    await assert.rejects(pending, { code: 'DOPPLER_GPU_DEVICE_LOST' });
    await assert.rejects(second.generateText(options), { code: 'DOPPLER_GPU_DEVICE_LOST' });
  } finally { await stream.return().catch(() => {}); await first.close(); await second.close(); }
  assert(device.buffers.every(buffer => buffer.destroyed));
});

test('device loss while a program is being constructed rejects opening and closes the late program', async () => {
  const { device, loss } = deviceFixture(), entered = deferred(), gate = deferred();
  const runtime = runtimeFixture(device, async () => { entered.resolve(); await gate.promise; });
  const opening = runtime.open(); await entered.promise;
  loss.resolve({ reason: 'unknown', message: 'lost while loading' }); await device.lost; gate.resolve();
  let opened;
  try { await assert.rejects(opening.then(value => { opened = value; }), { code: 'DOPPLER_GPU_DEVICE_LOST' }); }
  finally { await opened?.close(); }
  assert.equal(runtime.programs[0].closed, true);
});

test('failed buffer replacement cannot resurrect the released slot', () => {
  const { device } = deviceFixture(), binder = createResourceBinder(device);
  const layout = bytes => ({ bufferSlots: [{ slotId: 'input', owner: 'runtime', scope: 'session', usageBits: 8,
    size: { op: 'constant', bytes } }] });
  binder.bindSlots(layout(4)); device.failAllocation = true;
  assert.throws(() => binder.bindSlots(layout(8)), /allocation failure/); device.failAllocation = false;
  binder.bindSlots(layout(4)); binder.writeSlot('input', new Uint32Array([1])); binder.releaseAll();
});

test('a failed program-slot release still cleans other slots and closes the program', async () => {
  const { device } = deviceFixture(); let closed = 0, releases = 0;
  const program = { bindProgramSlot: () => ({}), releaseProgramSlot() { releases++; throw new Error('injected release failure'); },
    async close() { closed++; } };
  const binder = createResourceBinder(device, program);
  binder.bindSlots({ bufferSlots: ['program', 'runtime'].map((owner, index) => ({ slotId: `slot-${index}`, owner,
    scope: 'session', usageBits: 8, size: { op: 'constant', bytes: 4 } })) });
  const controller = createSessionController({}, binder, program);
  await assert.rejects(controller.close(), /release failure/);
  assert.equal(closed, 1); assert.equal(releases, 1); assert(device.buffers.every(buffer => buffer.destroyed));
  await controller.close(); binder.releaseAll(); assert.equal(releases, 1);
});
