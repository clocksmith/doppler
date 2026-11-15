/** Separate evaluator process: receives evaluation keys and BFV ciphertexts, never the client secret key. */
import { createHash } from 'node:crypto';
import { openExperimentContext } from './seal-context.js';
import { modelId } from './model.js';
import { createScoreEvaluation } from './evaluation.js';
import { envelope, fail, readFrames, validate, writeFrame } from './wire.js';
import policy from './policy.json' with { type: 'json' };
const runtime = await openExperimentContext();
const { seal, context, own } = runtime;
try {
  const frames = readFrames(process.stdin)[Symbol.asyncIterator]();
  const opening = await frames.next(), hello = opening.value;
  if (opening.done || typeof hello.session !== 'string' || !/^[a-f0-9-]{36}$/.test(hello.session) || typeof hello.keysId !== 'string') fail();
  const binding = envelope(hello.session, hello.keysId, -1, hello.integrityMode);
  const serialized = validate(hello, binding, 'evaluationKeys');
  if ('sha256:' + createHash('sha256').update(serialized).digest('hex') !== hello.keysId) fail();
  const keys = own(new seal.GaloisKeys()); keys.loadFromBase64(context, serialized);
  const evaluate = createScoreEvaluation(runtime, keys);
  // No acknowledgement containing any client content is needed.
  for (let step = 0; step < policy.steps; step++) {
    const frame = await frames.next(); if (frame.done) fail();
    const ciphertext = validate(frame.value, envelope(hello.session, hello.keysId, step, hello.integrityMode), 'inputCiphertext');
    await writeFrame(process.stdout, { ...envelope(hello.session, hello.keysId, step, hello.integrityMode), scoresCiphertext: evaluate(ciphertext) });
  }
  if (!(await frames.next()).done) fail();
  if (modelId !== binding.modelId) fail();
} catch { process.stderr.write('Encrypted experiment provider rejected a frame.\n'); process.exitCode = 1; }
finally { runtime.close(); }
