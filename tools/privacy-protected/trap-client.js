// The requester's trap-oracle client: one fresh, secret batch of known-answer traps per session, fetched and opened
// (signature, binding, expiry, decryption) before anything reaches an evaluator. The requester pins the oracle's
// Ed25519 key and the circuit identity; it loads no weights.

import { checkTrapLogits, circuitPin, createTrapRequest, openTrapBatch } from './traps.js';

export { checkTrapLogits };

export async function fetchTraps({ endpoint, oracleKeys, circuit, modelId, promptLength, generate, count, maxTolerance = 1, timeoutMs = 30000 }) {
  if (!oracleKeys?.length) throw new Error('--trap-oracle needs --oracle-key (the pinned oracle key)');
  if (!/^sha256:[a-f0-9]{64}$/.test(circuit ?? '')) throw new Error('--trap-oracle needs --circuit (the pinned circuit identity)');
  const { request, secret } = createTrapRequest({ circuit, promptLength, generate, count });
  const res = await fetch(new URL('/v1/trap-batch', endpoint), {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(request), signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`the trap oracle answered ${res.status}: ${(await res.text()).trim()}. Nothing was sent.`);
  const envelope = await res.json();
  const opened = openTrapBatch(envelope, { request, secret, oracleKeys, circuit: circuitPin(circuit, modelId), maxTolerance });
  return { batchId: opened.batchId, traps: opened.traps, tolerance: opened.tolerance.maxAbsLogitError, oracleKey: opened.oracleKey };
}
