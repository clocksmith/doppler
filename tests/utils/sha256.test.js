import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { sha256BytesHex, sha256Hex } from '../../src/utils/sha256.js';

const vectors = [
  '',
  'abc',
  'attention_streaming_f16kv.wgsl#main',
  'The quick brown fox jumps over the lazy dog',
  'The quick brown fox jumps over the lazy dog.',
  '你好，Doppler',
  '🙂 unicode and ascii mix',
];

for (const input of vectors) {
  const expected = createHash('sha256').update(input).digest('hex');
  const actual = sha256Hex(input);
  assert.equal(actual, expected, `sha256 mismatch for input: ${JSON.stringify(input)}`);
}

const byteVectors = [
  new Uint8Array(),
  new Uint8Array([0, 1, 2, 3, 255]),
  new Uint8Array(new Float32Array([1, -2.5, 0, Number.POSITIVE_INFINITY]).buffer),
];

for (const bytes of byteVectors) {
  const expected = createHash('sha256').update(bytes).digest('hex');
  const actual = sha256BytesHex(bytes);
  assert.equal(actual, expected, `sha256 byte mismatch for length: ${bytes.length}`);
}

console.log('sha256.test: ok');
