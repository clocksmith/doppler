const INITIAL_STATE = Object.freeze([
  0x6a09e667,
  0xbb67ae85,
  0x3c6ef372,
  0xa54ff53a,
  0x510e527f,
  0x9b05688c,
  0x1f83d9ab,
  0x5be0cd19,
]);

const ROUND_CONSTANTS = Object.freeze([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotateRight32(value, bits) {
  return (value >>> bits) | (value << (32 - bits));
}

function toSha256PaddedBytes(value) {
  const message = String(value ?? '');
  const encoder = new TextEncoder();
  const data = encoder.encode(message);
  const bitLength = BigInt(data.length) * 8n;
  const padBytes = (64 - ((data.length + 1 + 8) % 64)) % 64;
  const totalLength = data.length + 1 + padBytes + 8;
  const padded = new Uint8Array(totalLength);

  padded.set(data, 0);
  padded[data.length] = 0x80;

  for (let index = 0; index < 8; index += 1) {
    const shift = BigInt(56 - index * 8);
    padded[totalLength - 8 + index] = Number((bitLength >> shift) & 0xffn);
  }

  return padded;
}

function appendWordsFromBytes(words, bytes, offset) {
  for (let index = 0; index < 16; index += 1) {
    const byteOffset = offset + index * 4;
    words[index] = (
      (bytes[byteOffset] << 24)
      | (bytes[byteOffset + 1] << 16)
      | (bytes[byteOffset + 2] << 8)
      | bytes[byteOffset + 3]
    ) >>> 0;
  }
}

export function sha256Hex(value) {
  const padded = toSha256PaddedBytes(value);
  const hashState = INITIAL_STATE.slice();
  const schedule = new Uint32Array(64);

  for (let offset = 0; offset < padded.length; offset += 64) {
    appendWordsFromBytes(schedule, padded, offset);

    for (let index = 16; index < 64; index += 1) {
      const s0 = (
        rotateRight32(schedule[index - 15], 7)
        ^ rotateRight32(schedule[index - 15], 18)
        ^ (schedule[index - 15] >>> 3)
      ) >>> 0;
      const s1 = (
        rotateRight32(schedule[index - 2], 17)
        ^ rotateRight32(schedule[index - 2], 19)
        ^ (schedule[index - 2] >>> 10)
      ) >>> 0;
      schedule[index] = (
        schedule[index - 16]
        + s0
        + schedule[index - 7]
        + s1
      ) >>> 0;
    }

    let [a, b, c, d, e, f, g, h] = hashState;

    for (let index = 0; index < 64; index += 1) {
      const sigma1 = (
        rotateRight32(e, 6)
        ^ rotateRight32(e, 11)
        ^ rotateRight32(e, 25)
      ) >>> 0;
      const choose = ((e & f) ^ (~e & g)) >>> 0;
      const temp1 = (h + sigma1 + choose + ROUND_CONSTANTS[index] + schedule[index]) >>> 0;
      const sigma0 = (
        rotateRight32(a, 2)
        ^ rotateRight32(a, 13)
        ^ rotateRight32(a, 22)
      ) >>> 0;
      const majority = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const temp2 = (sigma0 + majority) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    hashState[0] = (hashState[0] + a) >>> 0;
    hashState[1] = (hashState[1] + b) >>> 0;
    hashState[2] = (hashState[2] + c) >>> 0;
    hashState[3] = (hashState[3] + d) >>> 0;
    hashState[4] = (hashState[4] + e) >>> 0;
    hashState[5] = (hashState[5] + f) >>> 0;
    hashState[6] = (hashState[6] + g) >>> 0;
    hashState[7] = (hashState[7] + h) >>> 0;
  }

  return hashState
    .map((value32) => value32.toString(16).padStart(8, '0'))
    .join('');
}
