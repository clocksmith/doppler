import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { authenticateQwen4ExpSourceBuffers, resolveQwen4ExpSourceContracts } from '../../src/converter/qwen4-exp-source.js';

// Contract: conversion accepts the checkpoint's I64 n-gram buffers only when
// they equal the authored contract exactly, and retains them with SHA-256 in
// manifest.conversion.sourceBuffers.
const prep = '../../artifacts/qwen38-flash-next-20261013/';
const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const flashNext = await readJson(`${prep}source/config.json`);
const headers = await readJson(`${prep}checkpoint-headers.json`);
const { ngramEmbedding } = resolveQwen4ExpSourceContracts(flashNext);

const encode = (values) => {
  const bytes = new Uint8Array(values.length * 8);
  const view = new DataView(bytes.buffer);
  values.forEach((value, i) => view.setBigInt64(i * 8, BigInt(value), true));
  return bytes;
};
const source = Object.fromEntries(Object.values(headers.buffers).map((buffer) => [buffer.tensor, encode(buffer.values)]));
const tensors = Object.keys(source).map((name) => ({ name, dtype: 'I64' }));
const read = async (tensor) => source[tensor.name];

const config = { inference: { ngramEmbedding }, manifest: { hashAlgorithm: 'sha256', conversion: { convertedAt: null } } };
const receipt = await authenticateQwen4ExpSourceBuffers(tensors, config, read);
assert.equal(receipt.length, 3);
for (const entry of receipt) {
  const recorded = Object.values(headers.buffers).find((buffer) => buffer.tensor === entry.tensor);
  assert.equal(entry.sha256, recorded.sha256, 'SHA-256 equals the bytes read at the pinned revision');
  assert.deepEqual(entry.values, recorded.values);
}
assert.deepEqual(config.manifest.conversion.sourceBuffers, receipt);
assert.equal(config.manifest.conversion.convertedAt, null, 'existing conversion fields are kept');

// A single differing value, a missing buffer and a wrong dtype fail.
const tampered = { ...source };
const multipliers = headers.buffers.layer_multipliers;
tampered[multipliers.tensor] = encode([...multipliers.values.slice(0, 2), (BigInt(multipliers.values[2]) + 2n).toString()]);
await assert.rejects(authenticateQwen4ExpSourceBuffers(tensors, { inference: { ngramEmbedding }, manifest: {} },
  async (tensor) => tampered[tensor.name]), /differs from the authored ngramEmbedding multipliers/);
await assert.rejects(authenticateQwen4ExpSourceBuffers(tensors.slice(1), { inference: { ngramEmbedding }, manifest: {} }, read),
  /lacks the retained n-gram buffer/);
await assert.rejects(authenticateQwen4ExpSourceBuffers(tensors.map((t) => ({ ...t, dtype: 'F32' })),
  { inference: { ngramEmbedding }, manifest: {} }, read), /must be I64/);
// Without a contract nothing is read or recorded.
const plain = { inference: {}, manifest: {} };
assert.equal(await authenticateQwen4ExpSourceBuffers(tensors, plain, async () => { throw new Error('unexpected read'); }), null);
assert.deepEqual(plain.manifest, {});
console.log('qwen4-exp-source-buffers.test: ok');
