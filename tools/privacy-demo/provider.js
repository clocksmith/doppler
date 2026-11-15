/** Separate evaluator process: receives evaluation keys and BFV ciphertexts, never the client secret key. */
import { createHash } from 'node:crypto';
import { openExperimentContext } from './seal-context.js';
import { counts, modelId } from './model.js';
import { envelope, fail, readFrames, validate, writeFrame } from './wire.js';
import policy from './policy.json' with { type: 'json' };
const runtime = await openExperimentContext();
const { seal, context, own } = runtime;
try {
  const frames = readFrames(process.stdin)[Symbol.asyncIterator]();
  const opening = await frames.next(), hello = opening.value;
  if (opening.done || typeof hello.session !== 'string' || !/^[a-f0-9-]{36}$/.test(hello.session) || typeof hello.keysId !== 'string') fail();
  const binding = envelope(hello.session, hello.keysId, -1);
  const serialized = validate(hello, binding, 'evaluationKeys');
  if ('sha256:' + createHash('sha256').update(serialized).digest('hex') !== hello.keysId) fail();
  const keys = own(new seal.GaloisKeys()); keys.loadFromBase64(context, serialized);
  const evaluator = own(new seal.Evaluator(context)), encoder = own(new seal.BatchEncoder(context));
  const weights = new BigInt64Array(encoder.slotCount()), n = policy.vocabulary.length;
  for (let row = 0; row < n; row++) for (let column = 0; column < n; column++) weights[row * n + column] = BigInt(counts[row][column]);
  const plainWeights = own(new seal.Plaintext()); encoder.encode(weights, plainWeights);
  const input = own(new seal.Ciphertext()), output = own(new seal.Ciphertext()), rotated = own(new seal.Ciphertext());
  // No acknowledgement containing any client content is needed.
  for (let step = 0; step < policy.steps; step++) {
    const frame = await frames.next(); if (frame.done) fail();
    const ciphertext = validate(frame.value, envelope(hello.session, hello.keysId, step), 'inputCiphertext');
    input.loadFromBase64(context, ciphertext);
    evaluator.multiplyPlain(input, plainWeights, output);
    for (const shift of policy.rotationSteps) { evaluator.rotateRows(output, shift, keys, rotated); evaluator.addInplace(output, rotated); }
    await writeFrame(process.stdout, { ...envelope(hello.session, hello.keysId, step), scoresCiphertext: output.saveToBase64(seal.ComprModeType.none) });
  }
  if (!(await frames.next()).done) fail();
  if (modelId !== binding.modelId) fail();
} catch { process.stderr.write('Encrypted experiment provider rejected a frame.\n'); process.exitCode = 1; }
finally { runtime.close(); }
