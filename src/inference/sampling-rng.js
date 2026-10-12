import { contractAssert as assert, count } from '../config/canonical-contract-codec.js';

// Random123 Philox4x32-10. BigInt preserves the entire unsigned product;
// extracting a high word from a Number multiplication would round it first.
/** @type {import('./sampling-rng.js').philox4x32} */
export function philox4x32(counter, key) {
  assert(counter.length === 4 && key.length === 2, 'Philox requires four counter and two key words.');
  for (const word of [...counter, ...key]) count(word, 'Philox word', 0xffffffff);
  let [c0, c1, c2, c3] = counter;
  let [k0, k1] = key;
  for (let round = 0; round < 10; round++) {
    const p0 = 0xd2511f53n * BigInt(c0), p1 = 0xcd9e8d57n * BigInt(c2);
    const hi0 = Number(p0 >> 32n), lo0 = Number(p0 & 0xffffffffn);
    const hi1 = Number(p1 >> 32n), lo1 = Number(p1 & 0xffffffffn);
    [c0, c1, c2, c3] = [(hi1 ^ c1 ^ k0) >>> 0, lo1, (hi0 ^ c3 ^ k1) >>> 0, lo0];
    if (round < 9) { k0 = (k0 + 0x9e3779b9) >>> 0; k1 = (k1 + 0xbb67ae85) >>> 0; }
  }
  return [c0, c1, c2, c3];
}

/** @type {import('./sampling-rng.js').samplingWord} */
export function samplingWord(seedHex64, tokenIndex, purpose, draw) {
  assert(typeof seedHex64 === 'string' && /^[a-f0-9]{16}$/.test(seedHex64), 'Philox seed must be 16 lowercase hexadecimal digits.');
  count(tokenIndex, 'semantic output index');
  count(purpose, 'random purpose', 3);
  count(draw, 'random draw', 0x3ffffffff);
  const counter = [tokenIndex >>> 0, Math.floor(tokenIndex / 0x100000000), purpose, Math.floor(draw / 4)];
  const key = [Number.parseInt(seedHex64.slice(8), 16), Number.parseInt(seedHex64.slice(0, 8), 16)];
  return philox4x32(counter, key)[draw % 4];
}

/** @type {import('./sampling-rng.js').samplingUniform} */
export function samplingUniform(seedHex64, tokenIndex, purpose, draw) {
  return samplingWord(seedHex64, tokenIndex, purpose, draw) / 0x100000000;
}
