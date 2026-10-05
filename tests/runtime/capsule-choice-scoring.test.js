import assert from 'node:assert/strict';
import { createDopplerRun } from '../../src/capsule-runtime.js';
import { createCapsuleStreamAccumulator } from '../../src/client/runtime/capsule-operation-stream.js';
import { validateTargetPlan } from '../../src/config/target-plan.js';
import { TEST_CAPSULE_AUTHORITY, TEST_CAPSULE_PUBLIC_KEY, createSignedCapsuleFixture } from '../helpers/capsule-v2-fixture.js';

const request = { prompt: 'Evaluate:', choices: [{ id: 'yes', label: ' A' }, { id: 'no', label: ' B' }], maxSeqLen: 8 };
let calls = 0, release, started, captured, invalid = false;
const ports = fixture => ({
  trustedSigners: { [TEST_CAPSULE_AUTHORITY]: TEST_CAPSULE_PUBLIC_KEY }, artifactStore: fixture.artifactStore,
  device: { getDevice: () => ({ createBuffer: () => ({ destroy() {} }), createCommandEncoder() {}, queue: { writeBuffer() {} } }),
    getProfile: () => ({ surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 }) },
  programFactory: async () => ({ executionGraphHash: fixture.capsule.program.executionGraphHash,
    getActiveAdapterIdentity: () => null, async close() {},
    async scoreChoices(input, control) {
      calls++; captured = input; started?.();
      if (release) await new Promise(resolve => { release = resolve; });
      control.signal.throwIfAborted();
      return { schema: 'doppler.choice-scores/v1', interpretation: 'next-token-logits', calibration: null,
        choices: input.choices.map((choice, index) => ({ ...choice, tokenId: index + 1, logit: index })),
        selectedId: invalid ? 'yes' : 'no', promptTokenCount: 1 };
    }
  })
});
const unqualifiedFixture = await createSignedCapsuleFixture();
const qualifiedFixture = await createSignedCapsuleFixture({ operation: 'scoreChoices' });
const unqualified = await createDopplerRun(ports(unqualifiedFixture)).openCapsule(unqualifiedFixture.capsule);
const session = await createDopplerRun(ports(qualifiedFixture)).openCapsule(qualifiedFixture.capsule);
try {
  await assert.rejects(unqualified.scoreChoices(request), /not qualified/);
  assert.equal(calls, 0, 'generation qualification cannot authorize decision execution');
  for (const schema of ['doppler.capsule-operation-request/v1', 'doppler.capsule-operation-request/v2']) {
    const operation = { schema, operation: { name: 'scoreChoices', version: 1 },
      input: { prompt: request.prompt, choices: request.choices }, options: { maxSeqLen: request.maxSeqLen },
      assignment: { attemptId: 'decision-1' }, limits: { maxInputBytes: 4096, maxOutputBytes: 4096, deadlineAt: Date.now() + 60000 } };
    const events = [];
    for await (const event of session.executeOperation(operation)) events.push(event);
    assert.equal(events.length, 1, 'scoring emits one completion and no generated prose');
    assert.equal(events[0].output.selectedId, 'no');
    assert.equal(events[0].receipt.operation.name, 'scoreChoices');
    assert.equal(events[0].receipt.targetPlanDigest, session.selectedTargetPlanDigest);
    if (schema.endsWith('/v2')) {
      const accumulator = createCapsuleStreamAccumulator(operation);
      accumulator.accept(events[0]);
      assert.deepEqual(accumulator.finish(), events[0]);
    }
  }
  invalid = true;
  await assert.rejects(session.scoreChoices(request), /selection does not match/);
  invalid = false;
  const mutable = structuredClone(request);
  const pending = session.scoreChoices(mutable);
  mutable.choices[0].label = 'mutated';
  await pending;
  assert.equal(captured.choices[0].label, request.choices[0].label);
  release = () => {};
  const hasStarted = new Promise(resolve => { started = resolve; });
  const controller = new AbortController();
  const cancelled = session.scoreChoices(request, { signal: controller.signal });
  const cancelledCheck = assert.rejects(cancelled, /cancel/);
  await hasStarted;
  controller.abort(new Error('cancel submitted scoring'));
  assert.throws(() => session.scoreChoices(request), /already active/);
  release(); release = null; started = null;
  await cancelledCheck;
  assert.equal((await session.scoreChoices(request)).selectedId, 'no');
  const plan = structuredClone(qualifiedFixture.capsule.targetPlans[0]);
  plan.qualification[0].generatedTokens = 1;
  assert.equal(validateTargetPlan(plan).ok, false, 'mixed qualification counts rejected');
  release = () => {};
  const startedClosing = new Promise(resolve => { started = resolve; });
  const closingOperation = session.scoreChoices(request);
  const closingCheck = assert.rejects(closingOperation, /closed|cancel/i);
  await startedClosing;
  let closedSettled = false;
  const closing = session.close().then(() => { closedSettled = true; });
  await Promise.resolve();
  assert.equal(closedSettled, false, 'close must wait for submitted scoring to settle');
  assert.throws(() => session.scoreChoices(request), /closed/);
  release(); release = null; started = null;
  await closingCheck;
  await closing;
} finally { await Promise.all([unqualified.close(), session.close()]); }
console.log('capsule-choice-scoring.test.js passed (signed fixture and injected program; no physical qualification claim)');
