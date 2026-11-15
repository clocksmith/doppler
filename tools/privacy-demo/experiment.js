/** Client-owned autoregression. Decrypt and select locally, then freshly encrypt every feedback step. */
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createHash, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import policy from './policy.json' with { type: 'json' };
import { openExperimentContext } from './seal-context.js';
import { choose, modelId, reference, transitionScores } from './model.js';
import { envelope, readFrames, validate, writeFrame } from './wire.js';
export async function runExperiment({ seed, integrityMode = policy.integrity.defaultMode, signal = null, observe = () => {}, transformResponse = value => value } = {}) {
  let token = policy.vocabulary.indexOf(seed);
  if (!policy.integrity.modes.includes(integrityMode)) throw Error('Unsupported experiment integrity mode');
  if (token < 0) throw Error('Fixture seed is outside the vocabulary');
  signal?.throwIfAborted();
  const began = performance.now(), runtime = await openExperimentContext();
  const { seal, context, own } = runtime;
  let provider, exited, timer, cancel;
  let clientBytes = 0, providerBytes = 0, output = '', minimumNoiseBits = Infinity;
  const stepMs = [], clientReferenceAudit = { checkedSteps: 0, elapsedMs: 0 };
  try {
    signal?.throwIfAborted();
    const generator = own(new seal.KeyGenerator(context)), secret = own(generator.secretKey()), publicKey = own(generator.createPublicKey());
    const keys = own(generator.createGaloisKeysWithSteps(Int32Array.from(policy.rotationSteps)));
    const serialized = keys.saveToBase64(seal.ComprModeType.none);
    const keysId = 'sha256:' + createHash('sha256').update(serialized).digest('hex'), session = randomUUID();
    const encoder = own(new seal.BatchEncoder(context)), encryptor = own(new seal.Encryptor(context, publicKey));
    const decryptor = own(new seal.Decryptor(context, secret)), plain = own(new seal.Plaintext()), cipher = own(new seal.Ciphertext());
    provider = spawn(process.execPath, [fileURLToPath(new URL('./provider.js', import.meta.url))], { stdio: ['pipe', 'pipe', 'pipe'], env: {} });
    exited = once(provider, 'exit');
    provider.stdin.on('error', () => {});
    // Diagnostics are deliberately constant and bounded; ciphertext/parser errors never enter a log.
    let stderrBytes = 0;
    provider.stderr.on('data', chunk => { stderrBytes += chunk.length; if (stderrBytes > 1024) provider.kill(); });
    cancel = () => provider.kill(); signal?.addEventListener('abort', cancel, { once: true });
    timer = setTimeout(() => provider.kill(), policy.timeoutMs);
    const send = async packet => { signal?.throwIfAborted(); observe({ direction: 'client', packet: structuredClone(packet) }); clientBytes += await writeFrame(provider.stdin, packet); };
    const responses = readFrames(provider.stdout)[Symbol.asyncIterator]();
    await send({ ...envelope(session, keysId, -1, integrityMode), evaluationKeys: serialized });
    const setupMs = performance.now() - began;
    for (let step = 0; step < policy.steps; step++) {
      signal?.throwIfAborted();
      const started = performance.now(), vector = new BigInt64Array(encoder.slotCount()), n = policy.vocabulary.length;
      for (let row = 0; row < n; row++) vector[row * n + token] = 1n;
      encoder.encode(vector, plain); encryptor.encrypt(plain, cipher);
      await send({ ...envelope(session, keysId, step, integrityMode), inputCiphertext: cipher.saveToBase64(seal.ComprModeType.none) });
      const received = await responses.next(); signal?.throwIfAborted();
      if (received.done) throw Error('Encrypted experiment provider ended early');
      providerBytes += Buffer.byteLength(JSON.stringify(received.value) + '\n');
      observe({ direction: 'provider', packet: structuredClone(received.value) });
      const checked = validate(transformResponse(structuredClone(received.value)), envelope(session, keysId, step, integrityMode), 'scoresCiphertext');
      cipher.loadFromBase64(context, checked);
      const noise = decryptor.invariantNoiseBudget(cipher);
      if (noise <= 0) throw Error('Encrypted experiment exhausted noise budget');
      minimumNoiseBits = Math.min(noise, minimumNoiseBits);
      decryptor.decrypt(cipher, plain);
      const slots = encoder.decodeBigInt64(plain), scores = policy.vocabulary.map((_value, row) => Number(slots[row * n]));
      if (integrityMode === 'client-reference') {
        const auditStarted = performance.now(), expected = transitionScores(token);
        if (scores.some((score, index) => score !== expected[index])) throw Error('Client reference audit rejected evaluator scores');
        clientReferenceAudit.checkedSteps++; clientReferenceAudit.elapsedMs += performance.now() - auditStarted;
      }
      token = choose(scores); output += policy.vocabulary[token]; stepMs.push(performance.now() - started);
    }
    provider.stdin.end();
    if (!(await responses.next()).done) throw Error('Unexpected encrypted experiment response');
    const [code] = await exited;
    if (code !== 0) throw Error('Encrypted experiment provider failed');
    const referenceStarted = performance.now(), expected = reference(seed), referenceElapsedMs = performance.now() - referenceStarted;
    return { schema: policy.schema, qualified: false, modelId, protocol: policy.protocol, output,
      matchesReference: output === expected, referenceElapsedMs, steps: policy.steps, setupMs, stepMs, elapsedMs: performance.now() - began,
      clientBytes, providerBytes, minimumNoiseBits, integrityMode, clientReferenceAudit, maxRssKiB: process.resourceUsage().maxRSS,
      adversary: policy.adversary, leakage: policy.leakage, integrityProvided: false, productionBackendAvailable: false };
  } finally {
    clearTimeout(timer); signal?.removeEventListener('abort', cancel);
    try {
      if (provider) { provider.stdin.destroy(); if (provider.exitCode === null) provider.kill(); await exited; }
    } finally { runtime.close(); }
  }
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try { console.log(JSON.stringify(await runExperiment({ seed: 'b' }), null, 2)); }
  catch { process.stderr.write('Encrypted autoregression experiment failed.\n'); process.exitCode = 1; }
}
