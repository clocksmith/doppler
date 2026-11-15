import assert from 'node:assert/strict';
import test from 'node:test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { runExperiment } from './experiment.js';
import { envelope, readFrames, validate } from './wire.js';
import { reference } from './model.js';
import policy from './policy.json' with { type: 'json' };

test('multiple encrypted feedback steps match the public reference with no plaintext fields sent to the evaluator', async () => {
  const shapes = [];
  for (const seed of ['b', 'e']) {
    const frames = [];
    const report = await runExperiment({ seed, observe: value => frames.push(value) });
    assert.equal(report.output, reference(seed)); assert.equal(report.matchesReference, true);
    assert.equal(report.qualified, false); assert.equal(report.productionBackendAvailable, false);
    assert.equal(report.steps, policy.steps); assert.ok(report.minimumNoiseBits > 0);
    const inputs = frames.filter(row => row.direction === 'client');
    assert.equal(inputs.length, policy.steps + 1);
    assert.equal(frames.filter(row => row.direction === 'provider').length, policy.steps);
    assert.equal(new Set(inputs.slice(1).map(row => row.packet.inputCiphertext)).size, policy.steps);
    for (const { packet } of frames) {
      const payloadField = packet.step === -1 ? 'evaluationKeys' : packet.inputCiphertext ? 'inputCiphertext' : 'scoresCiphertext';
      assert.deepEqual(Object.keys(packet).sort(), [...Object.keys(envelope(packet.session, packet.keysId, packet.step)), payloadField].sort());
      for (const key of ['seed', 'prompt', 'token', 'scores', 'secretKey', 'output', 'inputHash']) assert.equal(Object.hasOwn(packet, key), false);
    }
    shapes.push(frames.map(row => [row.direction, Object.keys(row.packet).sort(), Buffer.byteLength(JSON.stringify(row.packet))]));
  }
  assert.deepEqual(shapes[0], shapes[1]);
});
test('rejects wrong session/model/step, extra plaintext fields and corrupt ciphertext before continuing feedback', async () => {
  for (const change of [value => ({ ...value, session: 'other' }), value => ({ ...value, modelId: 'other' }),
    value => ({ ...value, step: 7 }), value => ({ ...value, plaintextSeed: 'forbidden' }), value => ({ ...value, scoresCiphertext: 'AAAA' })]) {
    const frames = [];
    await assert.rejects(runExperiment({ seed: 'b', observe: value => frames.push(value), transformResponse: change }));
    assert.equal(frames.filter(row => row.direction === 'client').length, 2);
  }
});
test('settles cancellation without another feedback message or returning a partial private output', async () => {
  const controller = new AbortController(), frames = [];
  await assert.rejects(runExperiment({ seed: 'b', signal: controller.signal, observe(value) {
    frames.push(value); if (value.direction === 'provider') controller.abort(Error('cancelled experiment'));
  } }));
  assert.equal(frames.filter(row => row.direction === 'client').length, 2);
  await assert.rejects(runExperiment({ seed: 'b', signal: controller.signal }));
});
test('demonstrates the missing malicious-computation guarantee with a valid substituted ciphertext', async () => {
  let inputCiphertext;
  const result = await runExperiment({ seed: 'b', observe({ packet }) { if (packet.inputCiphertext) inputCiphertext = packet.inputCiphertext; },
    transformResponse: packet => ({ ...packet, scoresCiphertext: inputCiphertext }) });
  assert.notEqual(result.output, reference('b')); assert.equal(result.matchesReference, false);
  assert.equal(result.integrityProvided, false); assert.equal(result.qualified, false);
});
test('bounds malformed framing and rejects envelopes carrying an extra key', async () => {
  async function* huge() { yield Buffer.alloc(policy.maxFrameBytes + 1); }
  async function* incomplete() { yield Buffer.from('{'); }
  for (const stream of [huge(), incomplete()]) await assert.rejects(async () => { for await (const _frame of readFrames(stream)) assert.fail(); });
  const binding = envelope('session', 'key', 0);
  assert.throws(() => validate({ ...binding, inputCiphertext: 'AAAA', secretKey: 'forbidden' }, binding, 'inputCiphertext'));
});

test('the evaluator rejects malformed client envelopes without reflecting private fields into logs', async () => {
  const child = spawn(process.execPath, [fileURLToPath(new URL('./provider.js', import.meta.url))], { stdio: ['pipe', 'pipe', 'pipe'], env: {} });
  const exit = once(child, 'exit');
  let stdout = '', stderr = '';
  child.stdout.on('data', chunk => { stdout += chunk; }); child.stderr.on('data', chunk => { stderr += chunk; });
  child.stdin.end(JSON.stringify({ session: '', secretKey: 'private-input-canary' }) + '\n');
  assert.equal((await exit)[0], 1); assert.equal(stdout, '');
  assert.equal(stderr, 'Encrypted experiment provider rejected a frame.\n');
});
