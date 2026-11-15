/** Bounded protocol-specific envelopes. No prompt, token, plaintext score or secret-key fields. */
import { createHash } from 'node:crypto';
import policy from './policy.json' with { type: 'json' };
import { modelId } from './model.js';
const policyId = 'sha256:' + createHash('sha256').update(JSON.stringify(policy)).digest('hex');
export const fail = () => { throw Error('Encrypted experiment frame rejected'); };
export function envelope(session, keysId, step) {
  return { schema: policy.schema, protocol: policy.protocol, policyId, modelId, session, keysId, step };
}
export function validate(value, binding, field) {
  const keys = [...Object.keys(binding), field].sort();
  if (!value || Object.keys(value).sort().join() !== keys.join() || Object.keys(binding).some(key => value[key] !== binding[key])
    || typeof value[field] !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(value[field]) || value[field].length > policy.maxFrameBytes) fail();
  return value[field];
}
export async function* readFrames(stream) {
  let pending = Buffer.alloc(0);
  for await (const chunk of stream) {
    if (pending.byteLength + chunk.byteLength > policy.maxFrameBytes) fail();
    pending = Buffer.concat([pending, chunk]);
    let newline;
    while ((newline = pending.indexOf(10)) >= 0) {
      const line = pending.subarray(0, newline); pending = pending.subarray(newline + 1);
      let value;
      try { value = JSON.parse(line.toString('utf8')); } catch { fail(); }
      yield value;
    }
  }
  if (pending.byteLength) fail();
}
export async function writeFrame(stream, value) {
  const frame = JSON.stringify(value) + '\n';
  if (Buffer.byteLength(frame) > policy.maxFrameBytes) fail();
  await new Promise((resolve, reject) => stream.write(frame, error => error ? reject(error) : resolve()));
  return Buffer.byteLength(frame);
}
