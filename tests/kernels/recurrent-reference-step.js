import assert from 'node:assert/strict';
import { decodeCapturedTensor, recurrentReference } from './recurrent-reference.js';

/** Test-only one-token view, retaining the captured layouts and model constants. */
export function recurrentReferenceStep(input, expected, token, initialState) {
  const p = input.params;
  assert(Number.isInteger(token) && token >= 0 && token < p.numTokens);
  const tensor = (role, data) => {
    const bytes = Buffer.from(Float32Array.from(data).buffer);
    return { role, bytes: bytes.length, data: bytes.toString('base64') };
  };
  const row = (record, role, stride, offset = 0) => {
    const values = decodeCapturedTensor(record, role);
    return values.subarray(offset + token * stride, offset + (token + 1) * stride);
  };
  const bOffset = p.abPacked ? p.bProjOffsetElements : 0;
  const b = new Float32Array(bOffset + p.numVHeads); b.set(row(input, 'b', p.numVHeads, bOffset), bOffset);
  const step = { ...input, params: { ...p, numTokens: 1 }, tensors: [
    ...input.tensors.filter(t => ['dtBias', 'aLog', 'normWeight'].includes(t.role)),
    tensor('a', row(input, 'a', p.numVHeads)), tensor('b', b),
    tensor('z', row(input, 'z', p.valueDim + (p.qkvzPacked ? p.convDim : 0))), tensor('recurrentState', initialState),
  ] };
  const output = { ...expected, params: step.params, tensors: [tensor('convOutput', row(expected, 'convOutput', p.convDim))] };
  return recurrentReference(step, output);
}
