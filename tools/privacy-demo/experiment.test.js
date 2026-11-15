import assert from 'node:assert/strict';
import test from 'node:test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';
import { runExperiment } from './experiment.js';
import { envelope, readFrames, validate } from './wire.js';
import { openExperimentContext } from './seal-context.js';
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
  const result = await runExperiment({ seed: 'b', integrityMode: 'none', observe({ packet }) { if (packet.inputCiphertext) inputCiphertext = packet.inputCiphertext; },
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

test('full client recomputation detects incorrect scores without claiming a cryptographic proof', async () => {
  const honest = await runExperiment({ seed: 'b', integrityMode: 'client-reference' });
  assert.equal(honest.matchesReference, true); assert.equal(honest.clientReferenceAudit.checkedSteps, policy.steps);
  assert.equal(honest.integrityProvided, false); assert.equal(honest.qualified, false);
  let inputCiphertext;
  await assert.rejects(runExperiment({ seed: 'b', integrityMode: 'client-reference',
    observe({ packet }) { if (packet.inputCiphertext) inputCiphertext = packet.inputCiphertext; },
    transformResponse: packet => ({ ...packet, scoresCiphertext: inputCiphertext }) }), /Client reference audit rejected/);
});
test('demonstrates selective-failure privacy leakage even when client reference integrity checks are enabled', async () => {
  const continuationCounts = [];
  for (const seed of ['b', 'c']) {
    const attacker = await openExperimentContext(), { seal, context, own } = attacker;
    const keys = own(new seal.GaloisKeys()), input = own(new seal.Ciphertext()), scores = own(new seal.Ciphertext()), rotated = own(new seal.Ciphertext());
    const evaluator = own(new seal.Evaluator(context));
    let sent = 0;
    try {
      const running = runExperiment({ seed, integrityMode: 'client-reference', observe({ direction, packet }) {
        if (direction !== 'client') return;
        sent++;
        if (packet.evaluationKeys) keys.loadFromBase64(context, packet.evaluationKeys);
        if (packet.inputCiphertext) input.loadFromBase64(context, packet.inputCiphertext);
      }, transformResponse(packet) {
        // Identical attack for both clients, using only public evaluation keys and wire ciphertexts.
        scores.loadFromBase64(context, packet.scoresCiphertext);
        evaluator.rotateRows(input, 1, keys, rotated); evaluator.addInplace(scores, rotated);
        return { ...packet, scoresCiphertext: scores.saveToBase64(seal.ComprModeType.none) };
      } });
      if (seed === 'b') await assert.rejects(running, /Client reference audit rejected/);
      else assert.equal((await running).matchesReference, true);
      continuationCounts.push(sent);
    } finally { attacker.close(); }
  }
  // A malicious evaluator observes continuation vs early stop, revealing a predicate of the private seed.
  assert.deepEqual(continuationCounts, [2, policy.steps + 1]);
});

test('public ciphertext replay rejects the same selective-failure attack for every seed before secret-key operations', async () => {
  const continuationCounts = [];
  for (const seed of policy.vocabulary) {
    const attacker = await openExperimentContext(), { seal, context, own } = attacker;
    const keys = own(new seal.GaloisKeys()), input = own(new seal.Ciphertext()), scores = own(new seal.Ciphertext()), rotated = own(new seal.Ciphertext());
    const evaluator = own(new seal.Evaluator(context));
    let sent = 0;
    try {
      await assert.rejects(runExperiment({ seed, integrityMode: 'ciphertext-replay', observe({ direction, packet }) {
        if (direction !== 'client') return;
        sent++;
        if (packet.evaluationKeys) keys.loadFromBase64(context, packet.evaluationKeys);
        if (packet.inputCiphertext) input.loadFromBase64(context, packet.inputCiphertext);
      }, transformResponse(packet) {
        scores.loadFromBase64(context, packet.scoresCiphertext);
        evaluator.rotateRows(input, 1, keys, rotated); evaluator.addInplace(scores, rotated);
        return { ...packet, scoresCiphertext: scores.saveToBase64(seal.ComprModeType.none) };
      } }), /Ciphertext replay rejected evaluator output before decryption/);
      continuationCounts.push(sent);
    } finally { attacker.close(); }
  }
  assert.deepEqual(continuationCounts, policy.vocabulary.map(() => 2));
});
test('replay checks all four encrypted circuits, rejects valid substitution and binds integrity mode', async () => {
  for (const seed of policy.vocabulary) {
    const report = await runExperiment({ seed, integrityMode: 'ciphertext-replay' });
    assert.equal(report.output, reference(seed)); assert.equal(report.ciphertextReplay.checkedSteps, policy.steps);
    assert.ok(report.ciphertextReplay.elapsedMs > 0); assert.ok(report.minimumNoiseBits > 0);
    assert.equal(report.productionBackendAvailable, false); assert.equal(report.integrityProvided, false);
  }
  let inputCiphertext;
  await assert.rejects(runExperiment({ seed: 'b', observe({ packet }) { if (packet.inputCiphertext) inputCiphertext = packet.inputCiphertext; },
    transformResponse: packet => ({ ...packet, scoresCiphertext: inputCiphertext }) }), /Ciphertext replay rejected/);
  await assert.rejects(runExperiment({ seed: 'b', transformResponse: packet => ({ ...packet, integrityMode: 'none' }) }), /frame rejected/);
  const unaudited = await runExperiment({ seed: 'b', integrityMode: 'none' });
  assert.equal(unaudited.minimumNoiseBits, null); assert.equal(unaudited.ciphertextReplay.checkedSteps, 0);
});
