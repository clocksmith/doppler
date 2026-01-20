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

const MSG_PERMUTATION = new Uint8Array([
  2, 6, 3, 10, 7, 0, 4, 13,
  1, 11, 12, 5, 9, 14, 15, 8,
]);

function toBytes(data) {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

function rotr(value, shift) {
  return (value >>> shift) | (value << (32 - shift));
}

function g(state, a, b, c, d, x, y) {
  state[a] = (state[a] + state[b] + x) >>> 0;
  state[d] = rotr(state[d] ^ state[a], 16);
  state[c] = (state[c] + state[d]) >>> 0;
  state[b] = rotr(state[b] ^ state[c], 12);
  state[a] = (state[a] + state[b] + y) >>> 0;
  state[d] = rotr(state[d] ^ state[a], 8);
  state[c] = (state[c] + state[d]) >>> 0;
  state[b] = rotr(state[b] ^ state[c], 7);
}

function permute(message) {
  const next = new Uint32Array(16);
  for (let i = 0; i < 16; i++) {
    next[i] = message[MSG_PERMUTATION[i]];
  }
  return next;
}

function blockWordsFromBytes(bytes, offset, length) {
  const words = new Uint32Array(16);
  for (let i = 0; i < length; i++) {
    words[i >> 2] |= bytes[offset + i] << ((i & 3) * 8);
  }
  return words;
}

function compress(cv, blockWords, counter, blockLen, flags) {
  const state = new Uint32Array(16);
  state.set(cv, 0);
  state.set(IV, 8);

  const counterLow = counter >>> 0;
  const counterHigh = Math.floor(counter / 0x100000000) >>> 0;

  state[12] ^= counterLow;
  state[13] ^= counterHigh;
  state[14] ^= blockLen;
  state[15] ^= flags;

  let message = blockWords;
  for (let round = 0; round < 7; round++) {
    g(state, 0, 4, 8, 12, message[0], message[1]);
    g(state, 1, 5, 9, 13, message[2], message[3]);
    g(state, 2, 6, 10, 14, message[4], message[5]);
    g(state, 3, 7, 11, 15, message[6], message[7]);
    g(state, 0, 5, 10, 15, message[8], message[9]);
    g(state, 1, 6, 11, 12, message[10], message[11]);
    g(state, 2, 7, 8, 13, message[12], message[13]);
    g(state, 3, 4, 9, 14, message[14], message[15]);

    if (round < 6) {
      message = permute(message);
    }
  }

  return state;
}

function chainingValue(state) {
  const cv = new Uint32Array(8);
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
  let cv = key;
  const blockCount = chunkLen === 0 ? 1 : Math.ceil(chunkLen / BLOCK_LEN);
  let output = null;

  for (let blockIndex = 0; blockIndex < blockCount; blockIndex++) {
    const blockOffset = blockIndex * BLOCK_LEN;
    const blockLen = chunkLen === 0
      ? 0
      : Math.min(BLOCK_LEN, chunkLen - blockOffset);
    const blockWords = blockWordsFromBytes(chunkBytes, blockOffset, blockLen);

    let flags = 0;
    if (blockIndex === 0) flags |= CHUNK_START;
    if (blockIndex === blockCount - 1) flags |= CHUNK_END;

    output = {
      inputCv: cv,
      blockWords,
      counter: chunkCounter,
      blockLen,
      flags,
    };

    const state = compress(cv, blockWords, chunkCounter, blockLen, flags);
    cv = chainingValue(state);
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
    this.lastChunkOutput = null;
  }

  update(data) {
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
    if (this.chunkLen > 0 || this.chunkCounter === 0) {
      const chunkBytes = this.chunkBuffer.subarray(0, this.chunkLen);
      this.#commitChunk(chunkBytes, this.chunkLen);
      this.chunkLen = 0;
    }

    let cv = null;
    let rootOutput = null;

    if (this.cvStack.length === 0) {
      rootOutput = this.lastChunkOutput;
    } else {
      cv = this.cvStack.pop().cv;
      while (this.cvStack.length > 0) {
        const left = this.cvStack.pop().cv;
        if (this.cvStack.length === 0) {
          rootOutput = parentOutput(left, cv, this.key);
        }
        cv = parentCv(left, cv, this.key);
      }
      if (!rootOutput) {
        rootOutput = this.lastChunkOutput;
      }
    }

    return outputBytes(rootOutput, OUT_LEN);
  }

  #commitChunk(chunkBytes, chunkLen) {
    const { cv, output } = createChunkOutput(chunkBytes, chunkLen, this.chunkCounter, this.key);
    this.lastChunkOutput = output;
    this.chunkCounter += 1;
    this.#pushCv(cv);
  }

  #pushCv(cv) {
    let level = 0;
    let current = cv;

    while (this.cvStack.length > 0 && this.cvStack[this.cvStack.length - 1].level === level) {
      const left = this.cvStack.pop().cv;
      current = parentCv(left, current, this.key);
      level += 1;
    }

    this.cvStack.push({ cv: current, level });
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
