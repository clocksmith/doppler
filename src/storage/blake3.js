const IV = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

const CHUNK_LEN = 1024;
const BLOCK_LEN = 64;
const OUT_LEN = 32;

const CHUNK_START = 1;
const CHUNK_END = 2;
const PARENT = 4;
const ROOT = 8;

const MESSAGE_SCHEDULE = Object.freeze([
  new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
  new Uint8Array([2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]),
  new Uint8Array([3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1]),
  new Uint8Array([10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6]),
  new Uint8Array([12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4]),
  new Uint8Array([9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7]),
  new Uint8Array([11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13]),
]);

function toBytes(data) {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

function blockWordsFromBytes(bytes, offset, length, words) {
  words.fill(0);
  for (let i = 0; i < length; i++) {
    words[i >> 2] |= bytes[offset + i] << ((i & 3) * 8);
  }
  return words;
}

function compress(cv, blockWords, counter, blockLen, flags, out = new Uint32Array(16)) {
  let v0 = cv[0];
  let v1 = cv[1];
  let v2 = cv[2];
  let v3 = cv[3];
  let v4 = cv[4];
  let v5 = cv[5];
  let v6 = cv[6];
  let v7 = cv[7];
  let v8 = IV[0];
  let v9 = IV[1];
  let v10 = IV[2];
  let v11 = IV[3];
  let v12 = IV[4] ^ (counter >>> 0);
  let v13 = IV[5] ^ (Math.floor(counter / 0x100000000) >>> 0);
  let v14 = IV[6] ^ blockLen;
  let v15 = IV[7] ^ flags;
  for (let round = 0; round < 7; round++) {
    const schedule = MESSAGE_SCHEDULE[round];
    v0 = (v0 + v4 + blockWords[schedule[0]]) >>> 0;
    v12 ^= v0;
    v12 = (v12 >>> 16) | (v12 << 16);
    v8 = (v8 + v12) >>> 0;
    v4 ^= v8;
    v4 = (v4 >>> 12) | (v4 << 20);
    v0 = (v0 + v4 + blockWords[schedule[1]]) >>> 0;
    v12 ^= v0;
    v12 = (v12 >>> 8) | (v12 << 24);
    v8 = (v8 + v12) >>> 0;
    v4 ^= v8;
    v4 = (v4 >>> 7) | (v4 << 25);
    v1 = (v1 + v5 + blockWords[schedule[2]]) >>> 0;
    v13 ^= v1;
    v13 = (v13 >>> 16) | (v13 << 16);
    v9 = (v9 + v13) >>> 0;
    v5 ^= v9;
    v5 = (v5 >>> 12) | (v5 << 20);
    v1 = (v1 + v5 + blockWords[schedule[3]]) >>> 0;
    v13 ^= v1;
    v13 = (v13 >>> 8) | (v13 << 24);
    v9 = (v9 + v13) >>> 0;
    v5 ^= v9;
    v5 = (v5 >>> 7) | (v5 << 25);
    v2 = (v2 + v6 + blockWords[schedule[4]]) >>> 0;
    v14 ^= v2;
    v14 = (v14 >>> 16) | (v14 << 16);
    v10 = (v10 + v14) >>> 0;
    v6 ^= v10;
    v6 = (v6 >>> 12) | (v6 << 20);
    v2 = (v2 + v6 + blockWords[schedule[5]]) >>> 0;
    v14 ^= v2;
    v14 = (v14 >>> 8) | (v14 << 24);
    v10 = (v10 + v14) >>> 0;
    v6 ^= v10;
    v6 = (v6 >>> 7) | (v6 << 25);
    v3 = (v3 + v7 + blockWords[schedule[6]]) >>> 0;
    v15 ^= v3;
    v15 = (v15 >>> 16) | (v15 << 16);
    v11 = (v11 + v15) >>> 0;
    v7 ^= v11;
    v7 = (v7 >>> 12) | (v7 << 20);
    v3 = (v3 + v7 + blockWords[schedule[7]]) >>> 0;
    v15 ^= v3;
    v15 = (v15 >>> 8) | (v15 << 24);
    v11 = (v11 + v15) >>> 0;
    v7 ^= v11;
    v7 = (v7 >>> 7) | (v7 << 25);
    v0 = (v0 + v5 + blockWords[schedule[8]]) >>> 0;
    v15 ^= v0;
    v15 = (v15 >>> 16) | (v15 << 16);
    v10 = (v10 + v15) >>> 0;
    v5 ^= v10;
    v5 = (v5 >>> 12) | (v5 << 20);
    v0 = (v0 + v5 + blockWords[schedule[9]]) >>> 0;
    v15 ^= v0;
    v15 = (v15 >>> 8) | (v15 << 24);
    v10 = (v10 + v15) >>> 0;
    v5 ^= v10;
    v5 = (v5 >>> 7) | (v5 << 25);
    v1 = (v1 + v6 + blockWords[schedule[10]]) >>> 0;
    v12 ^= v1;
    v12 = (v12 >>> 16) | (v12 << 16);
    v11 = (v11 + v12) >>> 0;
    v6 ^= v11;
    v6 = (v6 >>> 12) | (v6 << 20);
    v1 = (v1 + v6 + blockWords[schedule[11]]) >>> 0;
    v12 ^= v1;
    v12 = (v12 >>> 8) | (v12 << 24);
    v11 = (v11 + v12) >>> 0;
    v6 ^= v11;
    v6 = (v6 >>> 7) | (v6 << 25);
    v2 = (v2 + v7 + blockWords[schedule[12]]) >>> 0;
    v13 ^= v2;
    v13 = (v13 >>> 16) | (v13 << 16);
    v8 = (v8 + v13) >>> 0;
    v7 ^= v8;
    v7 = (v7 >>> 12) | (v7 << 20);
    v2 = (v2 + v7 + blockWords[schedule[13]]) >>> 0;
    v13 ^= v2;
    v13 = (v13 >>> 8) | (v13 << 24);
    v8 = (v8 + v13) >>> 0;
    v7 ^= v8;
    v7 = (v7 >>> 7) | (v7 << 25);
    v3 = (v3 + v4 + blockWords[schedule[14]]) >>> 0;
    v14 ^= v3;
    v14 = (v14 >>> 16) | (v14 << 16);
    v9 = (v9 + v14) >>> 0;
    v4 ^= v9;
    v4 = (v4 >>> 12) | (v4 << 20);
    v3 = (v3 + v4 + blockWords[schedule[15]]) >>> 0;
    v14 ^= v3;
    v14 = (v14 >>> 8) | (v14 << 24);
    v9 = (v9 + v14) >>> 0;
    v4 ^= v9;
    v4 = (v4 >>> 7) | (v4 << 25);
  }
  out[0] = v0; out[1] = v1; out[2] = v2; out[3] = v3;
  out[4] = v4; out[5] = v5; out[6] = v6; out[7] = v7;
  out[8] = v8; out[9] = v9; out[10] = v10; out[11] = v11;
  out[12] = v12; out[13] = v13; out[14] = v14; out[15] = v15;
  return out;
}

function chainingValue(state, cv = new Uint32Array(8)) {
  for (let i = 0; i < 8; i++) {
    cv[i] = (state[i] ^ state[i + 8]) >>> 0;
  }
  return cv;
}

function stateToBytes(state) {
  const out = new Uint8Array(64);
  for (let i = 0; i < 16; i++) {
    const value = state[i];
    const offset = i * 4;
    out[offset] = value & 0xff;
    out[offset + 1] = (value >>> 8) & 0xff;
    out[offset + 2] = (value >>> 16) & 0xff;
    out[offset + 3] = (value >>> 24) & 0xff;
  }
  return out;
}

function createChunkOutput(chunkBytes, chunkLen, chunkCounter, key) {
  const cv = key.slice();
  const blockWords = new Uint32Array(16);
  const state = new Uint32Array(16);
  const blockCount = chunkLen === 0 ? 1 : Math.ceil(chunkLen / BLOCK_LEN);
  let output = null;

  for (let blockIndex = 0; blockIndex < blockCount; blockIndex++) {
    const blockOffset = blockIndex * BLOCK_LEN;
    const blockLen = chunkLen === 0
      ? 0
      : Math.min(BLOCK_LEN, chunkLen - blockOffset);
    blockWordsFromBytes(chunkBytes, blockOffset, blockLen, blockWords);

    let flags = 0;
    if (blockIndex === 0) flags |= CHUNK_START;
    if (blockIndex === blockCount - 1) flags |= CHUNK_END;

    if (blockIndex === blockCount - 1) {
      // Only the final block survives this chunk. Its input must remain immutable
      // while the working chaining value is updated for the parent tree.
      output = { inputCv: cv.slice(), blockWords, counter: chunkCounter, blockLen, flags };
    }

    compress(cv, blockWords, chunkCounter, blockLen, flags, state);
    chainingValue(state, cv);
  }

  return { cv, output };
}

function parentOutput(leftCv, rightCv, key) {
  const blockWords = new Uint32Array(16);
  blockWords.set(leftCv, 0);
  blockWords.set(rightCv, 8);
  return {
    inputCv: key,
    blockWords,
    counter: 0,
    blockLen: BLOCK_LEN,
    flags: PARENT,
  };
}

function parentCv(leftCv, rightCv, key) {
  const output = parentOutput(leftCv, rightCv, key);
  const state = compress(output.inputCv, output.blockWords, output.counter, output.blockLen, output.flags);
  return chainingValue(state);
}

function outputBytes(output, outLen) {
  const result = new Uint8Array(outLen);
  let offset = 0;
  let counter = 0;

  while (offset < outLen) {
    const state = compress(
      output.inputCv,
      output.blockWords,
      counter,
      output.blockLen,
      output.flags | ROOT
    );
    const block = stateToBytes(state);
    const take = Math.min(outLen - offset, block.length);
    result.set(block.subarray(0, take), offset);
    offset += take;
    counter += 1;
  }

  return result;
}

class Blake3Hasher {
  constructor() {
    this.key = IV;
    this.chunkBuffer = new Uint8Array(CHUNK_LEN);
    this.chunkLen = 0;
    this.chunkCounter = 0;
    this.cvStack = [];
    this.finalized = null;
  }

  update(data) {
    if (this.finalized) {
      throw new Error('BLAKE3 update called after finalize.');
    }
    const bytes = toBytes(data);
    let offset = 0;

    while (offset < bytes.length) {
      const available = CHUNK_LEN - this.chunkLen;
      const take = Math.min(available, bytes.length - offset);
      this.chunkBuffer.set(bytes.subarray(offset, offset + take), this.chunkLen);
      this.chunkLen += take;
      offset += take;

      if (this.chunkLen === CHUNK_LEN) {
        this.#commitChunk(this.chunkBuffer, this.chunkLen);
        this.chunkLen = 0;
      }
    }
  }

  finalize() {
    if (this.finalized) {
      return this.finalized.slice(0);
    }
    if (this.chunkLen > 0 || this.chunkCounter === 0) {
      const chunkBytes = this.chunkBuffer.subarray(0, this.chunkLen);
      this.#commitChunk(chunkBytes, this.chunkLen);
      this.chunkLen = 0;
    }

    if (this.cvStack.length === 0) {
      throw new Error('BLAKE3 finalize called with no chunks.');
    }

    let right = this.cvStack.pop();
    while (this.cvStack.length > 0) {
      const left = this.cvStack.pop();
      const output = parentOutput(left.cv, right.cv, this.key);
      right = {
        cv: parentCv(left.cv, right.cv, this.key),
        output,
        level: left.level + 1,
      };
    }

    this.finalized = outputBytes(right.output, OUT_LEN);
    return this.finalized.slice(0);
  }

  #commitChunk(chunkBytes, chunkLen) {
    const { cv, output } = createChunkOutput(chunkBytes, chunkLen, this.chunkCounter, this.key);
    this.chunkCounter += 1;
    this.#pushCv(cv, output);
  }

  #pushCv(cv, output) {
    let level = 0;
    let current = cv;
    let currentOutput = output;

    while (this.cvStack.length > 0 && this.cvStack[this.cvStack.length - 1].level === level) {
      const left = this.cvStack.pop();
      currentOutput = parentOutput(left.cv, current, this.key);
      current = parentCv(left.cv, current, this.key);
      level += 1;
    }

    this.cvStack.push({ cv: current, output: currentOutput, level });
  }
}

export function createHasher() {
  return new Blake3Hasher();
}

export async function hash(data) {
  const hasher = new Blake3Hasher();
  hasher.update(data);
  return hasher.finalize();
}

if (typeof globalThis !== 'undefined') {
  if (!globalThis.blake3) {
    globalThis.blake3 = { hash, createHasher };
  }
}
