// Deterministic gzip, byte-compatible with reference zlib 1.2.x at level 9
// (windowBits 15, memLevel 8, default strategy). complexity.json records gzip
// sizes; Node's bundled zlib is a modified fork whose output differs, so the
// measurement uses this direct port of zlib's deflate_slow and trees.c instead.
import { crc32 } from 'node:zlib';

const W_SIZE = 1 << 15;
const W_MASK = W_SIZE - 1;
const WINDOW_SIZE = 2 * W_SIZE;
const HASH_SIZE = 1 << 15;
const HASH_MASK = HASH_SIZE - 1;
const HASH_SHIFT = 5;
const MIN_MATCH = 3;
const MAX_MATCH = 258;
const MIN_LOOKAHEAD = MAX_MATCH + MIN_MATCH + 1;
const MAX_DIST = W_SIZE - MIN_LOOKAHEAD;
const WIN_INIT = MAX_MATCH;
const TOO_FAR = 4096;
const GOOD_MATCH = 32;
const MAX_LAZY = 258;
const NICE_MATCH = 258;
const MAX_CHAIN = 4096;
const LIT_BUFSIZE = 1 << 14;
const MAX_SYMBOLS = LIT_BUFSIZE - 1;

const LENGTH_CODES = 29;
const LITERALS = 256;
const L_CODES = LITERALS + 1 + LENGTH_CODES;
const D_CODES = 30;
const BL_CODES = 19;
const HEAP_SIZE = 2 * L_CODES + 1;
const MAX_BITS = 15;
const MAX_BL_BITS = 7;
const END_BLOCK = 256;
const REP_3_6 = 16;
const REPZ_3_10 = 17;
const REPZ_11_138 = 18;
const EXTRA_LBITS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const EXTRA_DBITS = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

const LENGTH_CODE = new Uint8Array(256);
const BASE_LENGTH = new Int32Array(LENGTH_CODES);
const DIST_CODE = new Uint8Array(512);
const BASE_DIST = new Int32Array(D_CODES);
{
  let length = 0;
  let code;
  for (code = 0; code < LENGTH_CODES - 1; code++) {
    BASE_LENGTH[code] = length;
    for (let n = 0; n < (1 << EXTRA_LBITS[code]); n++) LENGTH_CODE[length++] = code;
  }
  LENGTH_CODE[length - 1] = code;
  let dist = 0;
  for (code = 0; code < 16; code++) {
    BASE_DIST[code] = dist;
    for (let n = 0; n < (1 << EXTRA_DBITS[code]); n++) DIST_CODE[dist++] = code;
  }
  dist >>= 7;
  for (; code < D_CODES; code++) {
    BASE_DIST[code] = dist << 7;
    for (let n = 0; n < (1 << (EXTRA_DBITS[code] - 7)); n++) DIST_CODE[256 + dist++] = code;
  }
}
const dCode = (dist) => (dist < 256 ? DIST_CODE[dist] : DIST_CODE[256 + (dist >> 7)]);

function reverseBits(code, length) {
  let result = 0;
  do { result |= code & 1; code >>= 1; result <<= 1; } while (--length > 0);
  return result >> 1;
}

function makeTree(size) {
  return { freq: new Int32Array(size), code: new Int32Array(size), dad: new Int32Array(size), len: new Int32Array(size) };
}

function genCodes(tree, maxCode, blCount) {
  const nextCode = new Int32Array(MAX_BITS + 1);
  let code = 0;
  for (let bits = 1; bits <= MAX_BITS; bits++) {
    code = (code + blCount[bits - 1]) << 1;
    nextCode[bits] = code;
  }
  for (let n = 0; n <= maxCode; n++) {
    const len = tree.len[n];
    if (len !== 0) tree.code[n] = reverseBits(nextCode[len]++, len);
  }
}

const STATIC_LTREE = makeTree(L_CODES + 2);
const STATIC_DTREE = makeTree(D_CODES);
{
  const blCount = new Int32Array(MAX_BITS + 1);
  let n = 0;
  while (n <= 143) { STATIC_LTREE.len[n++] = 8; blCount[8]++; }
  while (n <= 255) { STATIC_LTREE.len[n++] = 9; blCount[9]++; }
  while (n <= 279) { STATIC_LTREE.len[n++] = 7; blCount[7]++; }
  while (n <= 287) { STATIC_LTREE.len[n++] = 8; blCount[8]++; }
  genCodes(STATIC_LTREE, L_CODES + 1, blCount);
  for (n = 0; n < D_CODES; n++) { STATIC_DTREE.len[n] = 5; STATIC_DTREE.code[n] = reverseBits(n, 5); }
}

class BitWriter {
  constructor() { this.bytes = []; this.buffer = 0; this.valid = 0; }
  send(value, length) {
    this.buffer |= value << this.valid;
    this.valid += length;
    while (this.valid >= 8) { this.bytes.push(this.buffer & 255); this.buffer >>>= 8; this.valid -= 8; }
  }
  windup() {
    if (this.valid > 0) this.bytes.push(this.buffer & 255);
    this.buffer = 0; this.valid = 0;
  }
}

function deflateRaw(input) {
  const window = new Uint8Array(WINDOW_SIZE);
  const head = new Int32Array(HASH_SIZE);
  const prev = new Int32Array(W_SIZE);
  const ltree = makeTree(HEAP_SIZE);
  const dtree = makeTree(2 * D_CODES + 1);
  const bltree = makeTree(2 * BL_CODES + 1);
  const lDesc = { tree: ltree, stree: STATIC_LTREE, extra: EXTRA_LBITS, base: LITERALS + 1, elems: L_CODES, maxLength: MAX_BITS, maxCode: 0 };
  const dDesc = { tree: dtree, stree: STATIC_DTREE, extra: EXTRA_DBITS, base: 0, elems: D_CODES, maxLength: MAX_BITS, maxCode: 0 };
  const blDesc = { tree: bltree, stree: null, extra: EXTRA_BLBITS, base: 0, elems: BL_CODES, maxLength: MAX_BL_BITS, maxCode: 0 };
  const heap = new Int32Array(HEAP_SIZE);
  const depth = new Int32Array(HEAP_SIZE);
  const blCount = new Int32Array(MAX_BITS + 1);
  const symDist = new Int32Array(LIT_BUFSIZE);
  const symLc = new Int32Array(LIT_BUFSIZE);
  const out = new BitWriter();
  let symNext = 0;
  let optLen = 0;
  let staticLen = 0;
  let heapLen = 0;
  let heapMax = 0;
  let inputNext = 0;
  let strstart = 0;
  let blockStart = 0;
  let lookahead = 0;
  let insert = 0;
  let insH = 0;
  let matchLength = MIN_MATCH - 1;
  let prevLength = MIN_MATCH - 1;
  let matchStart = 0;
  let matchAvailable = false;
  let highWater = 0;

  function initBlock() {
    for (let n = 0; n < L_CODES; n++) ltree.freq[n] = 0;
    for (let n = 0; n < D_CODES; n++) dtree.freq[n] = 0;
    for (let n = 0; n < BL_CODES; n++) bltree.freq[n] = 0;
    ltree.freq[END_BLOCK] = 1;
    optLen = staticLen = 0;
    symNext = 0;
  }

  const smaller = (tree, n, m) => tree.freq[n] < tree.freq[m] || (tree.freq[n] === tree.freq[m] && depth[n] <= depth[m]);

  function pqdownheap(tree, k) {
    const v = heap[k];
    let j = k << 1;
    while (j <= heapLen) {
      if (j < heapLen && smaller(tree, heap[j + 1], heap[j])) j++;
      if (smaller(tree, v, heap[j])) break;
      heap[k] = heap[j];
      k = j;
      j <<= 1;
    }
    heap[k] = v;
  }

  function genBitlen(desc) {
    const { tree, stree, extra, base, maxLength, maxCode } = desc;
    let overflow = 0;
    for (let bits = 0; bits <= MAX_BITS; bits++) blCount[bits] = 0;
    tree.len[heap[heapMax]] = 0;
    let h;
    for (h = heapMax + 1; h < HEAP_SIZE; h++) {
      const n = heap[h];
      let bits = tree.len[tree.dad[n]] + 1;
      if (bits > maxLength) { bits = maxLength; overflow++; }
      tree.len[n] = bits;
      if (n > maxCode) continue;
      blCount[bits]++;
      const xbits = n >= base ? extra[n - base] : 0;
      const f = tree.freq[n];
      optLen += f * (bits + xbits);
      if (stree) staticLen += f * (stree.len[n] + xbits);
    }
    if (overflow === 0) return;
    do {
      let bits = maxLength - 1;
      while (blCount[bits] === 0) bits--;
      blCount[bits]--;
      blCount[bits + 1] += 2;
      blCount[maxLength]--;
      overflow -= 2;
    } while (overflow > 0);
    for (let bits = maxLength; bits !== 0; bits--) {
      let n = blCount[bits];
      while (n !== 0) {
        const m = heap[--h];
        if (m > maxCode) continue;
        if (tree.len[m] !== bits) {
          optLen += (bits - tree.len[m]) * tree.freq[m];
          tree.len[m] = bits;
        }
        n--;
      }
    }
  }

  function buildTree(desc) {
    const { tree, stree, elems } = desc;
    let maxCode = -1;
    heapLen = 0;
    heapMax = HEAP_SIZE;
    for (let n = 0; n < elems; n++) {
      if (tree.freq[n] !== 0) { heap[++heapLen] = maxCode = n; depth[n] = 0; } else tree.len[n] = 0;
    }
    while (heapLen < 2) {
      const node = heap[++heapLen] = maxCode < 2 ? ++maxCode : 0;
      tree.freq[node] = 1;
      depth[node] = 0;
      optLen--;
      if (stree) staticLen -= stree.len[node];
    }
    desc.maxCode = maxCode;
    for (let n = heapLen >> 1; n >= 1; n--) pqdownheap(tree, n);
    let node = elems;
    do {
      const n = heap[1];
      heap[1] = heap[heapLen--];
      pqdownheap(tree, 1);
      const m = heap[1];
      heap[--heapMax] = n;
      heap[--heapMax] = m;
      tree.freq[node] = tree.freq[n] + tree.freq[m];
      depth[node] = (depth[n] >= depth[m] ? depth[n] : depth[m]) + 1;
      tree.dad[n] = tree.dad[m] = node;
      heap[1] = node++;
      pqdownheap(tree, 1);
    } while (heapLen >= 2);
    heap[--heapMax] = heap[1];
    genBitlen(desc);
    genCodes(tree, maxCode, blCount);
  }

  function scanTree(tree, maxCode) {
    let prevlen = -1;
    let nextlen = tree.len[0];
    let count = 0;
    let maxCount = 7;
    let minCount = 4;
    if (nextlen === 0) { maxCount = 138; minCount = 3; }
    tree.len[maxCode + 1] = 0xffff;
    for (let n = 0; n <= maxCode; n++) {
      const curlen = nextlen;
      nextlen = tree.len[n + 1];
      if (++count < maxCount && curlen === nextlen) continue;
      else if (count < minCount) bltree.freq[curlen] += count;
      else if (curlen !== 0) {
        if (curlen !== prevlen) bltree.freq[curlen]++;
        bltree.freq[REP_3_6]++;
      } else if (count <= 10) bltree.freq[REPZ_3_10]++;
      else bltree.freq[REPZ_11_138]++;
      count = 0;
      prevlen = curlen;
      if (nextlen === 0) { maxCount = 138; minCount = 3; } else if (curlen === nextlen) { maxCount = 6; minCount = 3; } else { maxCount = 7; minCount = 4; }
    }
  }

  const sendCode = (c, tree) => out.send(tree.code[c], tree.len[c]);

  function sendTree(tree, maxCode) {
    let prevlen = -1;
    let nextlen = tree.len[0];
    let count = 0;
    let maxCount = 7;
    let minCount = 4;
    if (nextlen === 0) { maxCount = 138; minCount = 3; }
    for (let n = 0; n <= maxCode; n++) {
      const curlen = nextlen;
      nextlen = tree.len[n + 1];
      if (++count < maxCount && curlen === nextlen) continue;
      else if (count < minCount) {
        do { sendCode(curlen, bltree); } while (--count !== 0);
      } else if (curlen !== 0) {
        if (curlen !== prevlen) { sendCode(curlen, bltree); count--; }
        sendCode(REP_3_6, bltree); out.send(count - 3, 2);
      } else if (count <= 10) {
        sendCode(REPZ_3_10, bltree); out.send(count - 3, 3);
      } else {
        sendCode(REPZ_11_138, bltree); out.send(count - 11, 7);
      }
      count = 0;
      prevlen = curlen;
      if (nextlen === 0) { maxCount = 138; minCount = 3; } else if (curlen === nextlen) { maxCount = 6; minCount = 3; } else { maxCount = 7; minCount = 4; }
    }
  }

  function buildBlTree() {
    scanTree(ltree, lDesc.maxCode);
    scanTree(dtree, dDesc.maxCode);
    buildTree(blDesc);
    let maxBlindex;
    for (maxBlindex = BL_CODES - 1; maxBlindex >= 3; maxBlindex--) {
      if (bltree.len[BL_ORDER[maxBlindex]] !== 0) break;
    }
    optLen += 3 * (maxBlindex + 1) + 5 + 5 + 4;
    return maxBlindex;
  }

  function compressBlock(lt, dt) {
    for (let sx = 0; sx < symNext; sx++) {
      let dist = symDist[sx];
      let lc = symLc[sx];
      if (dist === 0) sendCode(lc, lt);
      else {
        let code = LENGTH_CODE[lc];
        sendCode(code + LITERALS + 1, lt);
        let extra = EXTRA_LBITS[code];
        if (extra !== 0) { lc -= BASE_LENGTH[code]; out.send(lc, extra); }
        dist--;
        code = dCode(dist);
        sendCode(code, dt);
        extra = EXTRA_DBITS[code];
        if (extra !== 0) { dist -= BASE_DIST[code]; out.send(dist, extra); }
      }
    }
    sendCode(END_BLOCK, lt);
  }

  function flushBlock(last) {
    const storedLen = strstart - blockStart;
    const hasBuffer = blockStart >= 0;
    buildTree(lDesc);
    buildTree(dDesc);
    const maxBlindex = buildBlTree();
    let optLenb = (optLen + 3 + 7) >>> 3;
    const staticLenb = (staticLen + 3 + 7) >>> 3;
    if (staticLenb <= optLenb) optLenb = staticLenb;
    if (storedLen + 4 <= optLenb && hasBuffer) {
      out.send(last ? 1 : 0, 3);
      out.windup();
      out.bytes.push(storedLen & 255, (storedLen >>> 8) & 255, ~storedLen & 255, (~storedLen >>> 8) & 255);
      for (let i = 0; i < storedLen; i++) out.bytes.push(window[blockStart + i]);
    } else if (staticLenb === optLenb) {
      out.send(2 + (last ? 1 : 0), 3);
      compressBlock(STATIC_LTREE, STATIC_DTREE);
    } else {
      out.send(4 + (last ? 1 : 0), 3);
      const lcodes = lDesc.maxCode + 1;
      const dcodes = dDesc.maxCode + 1;
      const blcodes = maxBlindex + 1;
      out.send(lcodes - 257, 5);
      out.send(dcodes - 1, 5);
      out.send(blcodes - 4, 4);
      for (let rank = 0; rank < blcodes; rank++) out.send(bltree.len[BL_ORDER[rank]], 3);
      sendTree(ltree, lcodes - 1);
      sendTree(dtree, dcodes - 1);
      compressBlock(ltree, dtree);
    }
    initBlock();
    if (last) out.windup();
    blockStart = strstart;
  }

  function tally(dist, lc) {
    symDist[symNext] = dist;
    symLc[symNext] = lc;
    symNext++;
    if (dist === 0) ltree.freq[lc]++;
    else {
      dist--;
      ltree.freq[LENGTH_CODE[lc] + LITERALS + 1]++;
      dtree.freq[dCode(dist)]++;
    }
    return symNext === MAX_SYMBOLS;
  }

  function slideHash() {
    for (let n = 0; n < HASH_SIZE; n++) { const m = head[n]; head[n] = m >= W_SIZE ? m - W_SIZE : 0; }
    for (let n = 0; n < W_SIZE; n++) { const m = prev[n]; prev[n] = m >= W_SIZE ? m - W_SIZE : 0; }
  }

  function fillWindow() {
    do {
      let more = WINDOW_SIZE - lookahead - strstart;
      if (strstart >= W_SIZE + MAX_DIST) {
        window.copyWithin(0, W_SIZE, W_SIZE + W_SIZE - more);
        matchStart -= W_SIZE;
        strstart -= W_SIZE;
        blockStart -= W_SIZE;
        if (insert > strstart) insert = strstart;
        slideHash();
        more += W_SIZE;
      }
      if (inputNext >= input.length) break;
      const n = Math.min(more, input.length - inputNext);
      window.set(input.subarray(inputNext, inputNext + n), strstart + lookahead);
      inputNext += n;
      lookahead += n;
      if (lookahead + insert >= MIN_MATCH) {
        let str = strstart - insert;
        insH = window[str];
        insH = ((insH << HASH_SHIFT) ^ window[str + 1]) & HASH_MASK;
        while (insert) {
          insH = ((insH << HASH_SHIFT) ^ window[str + MIN_MATCH - 1]) & HASH_MASK;
          prev[str & W_MASK] = head[insH];
          head[insH] = str;
          str++;
          insert--;
          if (lookahead + insert < MIN_MATCH) break;
        }
      }
    } while (lookahead < MIN_LOOKAHEAD && inputNext < input.length);
    if (highWater < WINDOW_SIZE) {
      const curr = strstart + lookahead;
      if (highWater < curr) {
        const init = Math.min(WINDOW_SIZE - curr, WIN_INIT);
        window.fill(0, curr, curr + init);
        highWater = curr + init;
      } else if (highWater < curr + WIN_INIT) {
        const init = Math.min(curr + WIN_INIT - highWater, WINDOW_SIZE - highWater);
        window.fill(0, highWater, highWater + init);
        highWater += init;
      }
    }
  }

  function insertString(str) {
    insH = ((insH << HASH_SHIFT) ^ window[str + MIN_MATCH - 1]) & HASH_MASK;
    const matchHead = prev[str & W_MASK] = head[insH];
    head[insH] = str;
    return matchHead;
  }

  function longestMatch(curMatch) {
    let chainLength = MAX_CHAIN;
    const scanStart = strstart;
    let bestLen = prevLength;
    let niceMatch = NICE_MATCH;
    const limit = strstart > MAX_DIST ? strstart - MAX_DIST : 0;
    const strend = strstart + MAX_MATCH;
    let scanEnd1 = window[scanStart + bestLen - 1];
    let scanEnd = window[scanStart + bestLen];
    if (prevLength >= GOOD_MATCH) chainLength >>>= 2;
    if (niceMatch > lookahead) niceMatch = lookahead;
    do {
      let match = curMatch;
      if (window[match + bestLen] !== scanEnd || window[match + bestLen - 1] !== scanEnd1
        || window[match] !== window[scanStart] || window[++match] !== window[scanStart + 1]) continue;
      let scan = scanStart + 2;
      match++;
      // Eight comparisons per bound check, as in zlib; offset 2 is implied by the hash.
      while (window[++scan] === window[++match] && window[++scan] === window[++match]
        && window[++scan] === window[++match] && window[++scan] === window[++match]
        && window[++scan] === window[++match] && window[++scan] === window[++match]
        && window[++scan] === window[++match] && window[++scan] === window[++match]
        && scan < strend);
      const len = MAX_MATCH - (strend - scan);
      if (len > bestLen) {
        matchStart = curMatch;
        bestLen = len;
        if (len >= niceMatch) break;
        scanEnd1 = window[scanStart + bestLen - 1];
        scanEnd = window[scanStart + bestLen];
      }
    } while ((curMatch = prev[curMatch & W_MASK]) > limit && --chainLength !== 0);
    return bestLen <= lookahead ? bestLen : lookahead;
  }

  initBlock();
  for (;;) {
    if (lookahead < MIN_LOOKAHEAD) {
      fillWindow();
      if (lookahead === 0) break;
    }
    let hashHead = 0;
    if (lookahead >= MIN_MATCH) hashHead = insertString(strstart);
    prevLength = matchLength;
    const prevMatch = matchStart;
    matchLength = MIN_MATCH - 1;
    if (hashHead !== 0 && prevLength < MAX_LAZY && strstart - hashHead <= MAX_DIST) {
      matchLength = longestMatch(hashHead);
      if (matchLength <= 5 && matchLength === MIN_MATCH && strstart - matchStart > TOO_FAR) matchLength = MIN_MATCH - 1;
    }
    if (prevLength >= MIN_MATCH && matchLength <= prevLength) {
      const maxInsert = strstart + lookahead - MIN_MATCH;
      const flush = tally(strstart - 1 - prevMatch, prevLength - MIN_MATCH);
      lookahead -= prevLength - 1;
      prevLength -= 2;
      do {
        if (++strstart <= maxInsert) insertString(strstart);
      } while (--prevLength !== 0);
      matchAvailable = false;
      matchLength = MIN_MATCH - 1;
      strstart++;
      if (flush) flushBlock(false);
    } else if (matchAvailable) {
      if (tally(0, window[strstart - 1])) flushBlock(false);
      strstart++;
      lookahead--;
    } else {
      matchAvailable = true;
      strstart++;
      lookahead--;
    }
  }
  if (matchAvailable) { tally(0, window[strstart - 1]); matchAvailable = false; }
  flushBlock(true);
  return Uint8Array.from(out.bytes);
}

// gzip member as written by zlib with windowBits 31 at level 9, OS byte 255 and mtime 0.
export function gzipLevel9(data) {
  const input = data instanceof Uint8Array ? data : new Uint8Array(data);
  const body = deflateRaw(input);
  const result = new Uint8Array(10 + body.length + 8);
  result.set([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 2, 255]);
  result.set(body, 10);
  const view = new DataView(result.buffer);
  view.setUint32(10 + body.length, crc32(input), true);
  view.setUint32(14 + body.length, input.length >>> 0, true);
  return result;
}
