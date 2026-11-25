// Three direct-code backends for the same nineteen-instruction Sprout.
//
// Called by build-compiler.js after its JS/WGSL templates. This bootstrap is not
// needed by any descendant. Binary templates describe instructions and generic
// runtime helpers, never a precompiled copy of the compiler.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { asciiBytes, jsonText, readText, writeText } from './build-support.js';
import { gzipLevel9 } from './zlib-gzip.js';

// Replace the single match of `pattern`; `replacement` receives the match groups.
function replaceOnce(pattern, replacement, value) {
  const global = new RegExp(pattern.source, pattern.flags.includes('g') ? pattern.flags : pattern.flags + 'g');
  let count = 0;
  const result = value.replace(global, (...match) => {
    count++;
    return typeof replacement === 'function' ? replacement(...match) : replacement;
  });
  if (count !== 1) throw new Error(`Expected one match of ${pattern}, found ${count}.`);
  return result;
}

// Byte sequences are plain arrays of numbers; strings contribute their ASCII bytes.
function bytes(...parts) {
  const result = [];
  for (const part of parts) {
    if (typeof part === 'number') result.push(part);
    else if (typeof part === 'string') result.push(...asciiBytes(part));
    else for (const byte of part) result.push(byte);
  }
  return result;
}

function uleb(n) {
  const result = [];
  while (n >= 128) {
    result.push((n % 128) | 128);
    n = Math.floor(n / 128);
  }
  result.push(n);
  return result;
}

function sleb(n) {
  n = n < 0x80000000 ? n : n - 0x100000000;
  const result = [];
  for (;;) {
    const part = n & 127;
    n = Math.floor(n / 128);
    const done = (n === 0 && !(part & 64)) || (n === -1 && (part & 64));
    result.push(done ? part : part | 128);
    if (done) return result;
  }
}

function padded(n, signed = false) {
  const top = Math.floor(n / 2 ** 28);
  return [...[0, 7, 14, 21].map((shift) => (Math.floor(n / 2 ** shift) & 127) | 128), top | (signed && top >= 8 ? 112 : 0)];
}

const section = (kind, payload) => bytes(kind, uleb(payload.length), payload);
const get = (n) => bytes(0x20, uleb(n));
const set = (n) => bytes(0x21, uleb(n));
const constant = (n) => bytes(0x41, sleb(n));
const binary = (left, right, opcode) => bytes(left, right, opcode);
const ifBlock = (condition, yes, no = null, result = false) => bytes(condition, 4, result ? 127 : 64, yes, no !== null ? bytes(0x05, no) : [], 0x0b);
const whileBlock = (condition, body) => bytes(0x02, 0x40, 0x03, 0x40, condition, 0x45, 0x0d, 0x01, body, 0x0c, 0x00, 0x0b, 0x0b);
const trapIf = (condition) => ifBlock(condition, [0x00]);
const load = (address) => bytes(address, 0x28, 0x02, 0x00);
const storeWord = (address, value) => bytes(address, value, 0x36, 0x02, 0x00);
function functionBody(localCount, body) {
  const locals = localCount ? bytes(0x01, uleb(localCount), 0x7f) : [0x00];
  const full = bytes(locals, body, 0x0b);
  return bytes(uleb(full.length), full);
}

const OUT = 262144;
const INPUT = 524288;
const STACK = 786432;
const [ADD, SUB, MUL, DIV, REM] = [0x6a, 0x6b, 0x6c, 0x6e, 0x70];
const [EQ, NE, LT, GT, LE, GE, AND, OR] = [0x46, 0x47, 0x49, 0x4b, 0x4d, 0x4f, 0x71, 0x72];

const putBody = bytes(
  trapIf(binary(get(0), constant(255), GT)),
  trapIf(binary([0x23, 0x00], constant(OUT + 262143), GE)),
  0x23, 0x00, get(0), 0x3a, 0x00, 0x00,
  0x23, 0x00, constant(1), 0x6a, 0x24, 0x00);
const decimalBody = bytes(constant(1), set(1), whileBlock(
  binary(binary(get(0), get(1), DIV), constant(10), GE),
  bytes(binary(get(1), constant(10), MUL), set(1))), whileBlock(get(1), bytes(
  binary(binary(get(0), get(1), DIV), constant(48), ADD), 0x10, 0x00,
  binary(get(0), get(1), REM), set(0),
  binary(get(1), constant(10), DIV), set(1))));

// Validator parameters: target, pointer, length. Locals: pc, data length,
// code length, opcode, a, b, c, depth, top. It inspects but never executes code.
const word = (index) => load(binary(get(1), binary(index, constant(4), MUL), ADD));
const topAddress = binary(constant(STACK), binary(binary(get(10), constant(1), SUB), constant(4), MUL), ADD);
const pushAddress = binary(constant(STACK), binary(get(10), constant(4), MUL), ADD);
const eqOp = (n) => binary(get(6), constant(n), EQ);
const isBetween = (reg, lo, hi) => binary(binary(get(reg), constant(lo), GE), binary(get(reg), constant(hi), LE), AND);
let validateBody = bytes(
  trapIf(binary(get(0), constant(2), GT)),
  trapIf(binary(binary(get(1), constant(0), NE), binary(get(1), constant(INPUT), NE), AND)),
  trapIf(binary(binary(get(2), constant(3), LT), binary(get(2), constant(60000), GT), OR)),
  trapIf(binary(word(constant(0)), constant(21331), NE)),
  word(constant(1)), set(4), word(constant(2)), set(5),
  trapIf(binary(binary(get(4), get(2), GT), binary(get(5), get(2), GT), OR)),
  trapIf(binary(get(5), constant(4), REM)),
  trapIf(binary(binary(binary(constant(3), get(4), ADD), get(5), ADD), get(2), NE)),
  binary(constant(3), get(4), ADD), set(3));
let validateLoop = bytes(...[0, 1, 2, 3].map((i) => bytes(word(binary(get(3), constant(i), ADD)), set(6 + i))));
validateLoop = bytes(validateLoop,
  trapIf(binary(binary(get(6), constant(18), GT), binary(get(7), constant(64), GE), OR)),
  trapIf(binary(isBetween(6, 2, 11), binary(get(8), constant(64), GE), AND)),
  trapIf(binary(isBetween(6, 3, 9), binary(get(9), constant(64), GE), AND)),
  ifBlock(binary(eqOp(14), eqOp(16), OR),
    bytes(trapIf(binary(get(10), constant(64), GE)),
      storeWord(pushAddress, get(6)), binary(get(10), constant(1), ADD), set(10)),
    ifBlock(eqOp(17),
      bytes(trapIf(bytes(get(10), 0x45)), trapIf(binary(load(topAddress), constant(16), NE)),
        storeWord(topAddress, constant(17))),
      ifBlock(binary(eqOp(15), eqOp(18), OR),
        bytes(trapIf(bytes(get(10), 0x45)), load(topAddress), set(11),
          ifBlock(eqOp(15), trapIf(binary(get(11), constant(14), NE)),
            trapIf(binary(binary(get(11), constant(16), NE), binary(get(11), constant(17), NE), AND))),
          binary(get(10), constant(1), SUB), set(10))))),
  binary(get(3), constant(4), ADD), set(3));
validateBody = bytes(validateBody, whileBlock(binary(get(3), get(2), LT), validateLoop), trapIf(get(10)));

const types = [0x03, 0x60, 0x01, 0x7f, 0x00, 0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x00, 0x60, 0x03, 0x7f, 0x7f, 0x7f, 0x01, 0x7f];
const globals = bytes(0x03, ...[[1, OUT], [0, OUT], [0, INPUT]].map(([mutable, value]) => bytes(127, mutable, constant(value), 0x0b)));
const exports = bytes(0x04, ...[['memory', 2, 0], ['run', 0, 3], ['output', 3, 1], ['input', 3, 2]]
  .map(([name, kind, index]) => bytes(uleb(name.length), name, kind, index)));
const prefix = bytes(0x00, 'asm', 0x01, 0x00, 0x00, 0x00, section(1, types), section(3, [0x04, 0x00, 0x00, 0x01, 0x02]),
  section(5, [0x01, 0x00, 0x10]), section(6, globals), section(7, exports), 0x0a);
const helpers = bytes(0x04, functionBody(0, putBody), functionBody(1, decimalBody), functionBody(9, validateBody));
const WASM_PARTS = [prefix, helpers,
  bytes(0x01, 0x40, 0x7f, get(1), 0x45, 0x04, 0x40, 0x41),
  bytes(set(2), 0x0b, constant(OUT), 0x24, 0x00, get(0), get(1), get(2), 0x10, 0x02, get(0), set(3), get(2), set(4)),
  bytes(0x23, 0x00, constant(OUT), 0x6b, 0x0b)];

// 0xff is a builder marker, never a Wasm opcode here: a/b/c are native local
// indices; i is an immediate; n is the embedded program's word count.
const reg = (letter) => bytes(0xff, letter);
const rg = (letter) => bytes(0x20, reg(letter));
const rs = (letter) => bytes(0x21, reg(letter));
const WASM_OP = [
  [0x01], bytes(0x41, reg('i'), rs('a')), bytes(rg('b'), rs('a')),
  ...[ADD, SUB, MUL].map((op) => bytes(binary(rg('b'), rg('c'), op), rs('a'))),
  ...[DIV, REM].map((op) => bytes(ifBlock(rg('c'), binary(rg('b'), rg('c'), op), constant(0), true), rs('a'))),
  ...[EQ, LT].map((op) => bytes(binary(rg('b'), rg('c'), op), rs('a'))),
  bytes(ifBlock(binary(rg('b'), bytes(0x41, reg('n')), LT),
    load(binary(rg('b'), constant(4), MUL)), constant(0), true), rs('a')),
  bytes(ifBlock(binary(rg('b'), get(2), LT),
    load(binary(get(1), binary(rg('b'), constant(4), MUL), ADD)), constant(0), true), rs('a')),
  bytes(rg('a'), 0x10, 0x00), bytes(rg('a'), 0x10, 0x01),
  bytes(0x02, 0x40, 0x03, 0x40, rg('a'), 0x45, 0x0d, 0x01),
  [0x0c, 0x00, 0x0b, 0x0b], bytes(rg('a'), 0x04, 0x40), [0x05], [0x0b]];

function expandWasm(template, operands, length) {
  const result = [];
  let i = 0;
  while (i < template.length) {
    const byte = template[i++];
    if (byte !== 255) {
      result.push(byte);
      continue;
    }
    const marker = String.fromCharCode(template[i++]);
    if ('abc'.includes(marker)) result.push(operands[marker.charCodeAt(0) - 97] + 3);
    else result.push(...padded(marker === 'i' ? operands[1] : length, true));
  }
  return result;
}

function packedWords(words) {
  let result = '';
  for (let value of words) {
    for (;;) {
      let digit = value % 32;
      value = Math.floor(value / 32);
      if (value) digit += 32;
      result += String.fromCharCode(48 + digit + (digit >= 44 ? 1 : 0));
      if (!value) break;
    }
  }
  return result;
}

function littleEndianWords(words) {
  const result = [];
  for (const value of words) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) throw new Error(`Not a u32 word: ${value}`);
    result.push(value & 255, (value >>> 8) & 255, (value >>> 16) & 255, value >>> 24);
  }
  return result;
}

function assemblyText(p, lines) {
  const chunks = [];
  for (let i = 0; i < p[1]; i += 32) chunks.push(p.slice(3 + i, 3 + Math.min(i + 32, p[1])).join(' '));
  return '.data\n' + chunks.join('\n') + '\n.code\n' + lines.join('\n') + '\n';
}

const EXAMPLES = {
  fibonacci: ['6765\n', [['set', 2, 20], ['set', 3, 0], ['set', 4, 1], ['set', 5, 1],
    ['while', 2], ['add', 6, 3, 4], ['mov', 3, 4], ['mov', 4, 6], ['sub', 2, 2, 5],
    ['end'], ['decimal', 3], ['set', 7, 10], ['put', 7]]],
  factorial: ['3628800\n', [['set', 2, 10], ['set', 3, 1], ['set', 4, 1], ['while', 2],
    ['mul', 3, 3, 2], ['sub', 2, 2, 4], ['end'], ['decimal', 3], ['set', 5, 10], ['put', 5]]],
  hello: ['Hello from Sprout!\n', [['set', 2, 3], ['set', 3, 3 + 'Hello from Sprout!\n'.length],
    ['set', 4, 1], ['lt', 5, 2, 3], ['while', 5], ['data', 6, 2], ['put', 6],
    ['add', 2, 2, 4], ['lt', 5, 2, 3], ['end']]],
  branch: ['Y\n', [['set', 2, 7], ['set', 3, 7], ['eq', 4, 2, 3], ['if', 4],
    ['set', 5, 89], ['else'], ['set', 5, 78], ['fi'], ['put', 5], ['set', 5, 10], ['put', 5]]],
  overflow: ['0\n', [['set', 2, 4294967295], ['set', 3, 1], ['add', 4, 2, 3],
    ['decimal', 4], ['set', 5, 10], ['put', 5]]],
};

export function buildWasm({ root, host, printJs, autorun, numbers, OPS, OPNUM, JS_OP, WG_OP, JS_PARTS, WG_PARTS, WG_RUNTIME }) {
  // Only compiler descendants need binary output. The other four constructions
  // continue to embed the original, strictly ASCII WebGPU host unchanged.
  let compilerHost = replaceOnce(/function decodeWords\(words\)/, 'function decodeWords(words, binary = false)', host);
  compilerHost = replaceOnce(/code\s*>\s*127/, 'code > (binary ? 255 : 127)', compilerHost);
  compilerHost = replaceOnce(
    /return new TextDecoder\("utf-8",\s*\{\s*fatal:\s*true\s*\}\)\.decode\(bytes\);/,
    'return binary ? bytes : new TextDecoder("utf-8", { fatal: true }).decode(bytes);',
    compilerHost);
  compilerHost = replaceOnce(/(async function dispatchText\([^)]*input\s*=\s*null)\)/,
    (_, head) => head + ', binary = false)', compilerHost);
  compilerHost = replaceOnce(/return\s*\{\s*text:\s*decodeWords\(words\),\s*words\s*\};/,
    'return binary ? { bytes: decodeWords(words, true), words } : { text: decodeWords(words), words };',
    compilerHost);

  JS_PARTS[2] = JS_PARTS[2].replace('function execute(', 'function executeBytes(');
  JS_PARTS[2] = replaceOnce(/target\s*!==\s*0\s*&&\s*target\s*!==\s*1/, 'target!==0&&target!==1&&target!==2', JS_PARTS[2]);
  JS_PARTS[2] = JS_PARTS[2].replaceAll('Use target 0 (JS) or 1 (WGSL).', 'Use target 0 (JS), 1 (WGSL), or 2 (Wasm).');
  JS_PARTS[2] = replaceOnce(/c\s*>\s*127/, 'c>255', JS_PARTS[2]);
  // Quote Sprout words, not compiled code, with five payload bits per character.
  // ASCII 48..112 excluding backslash needs no escaping in a JS string. The high
  // half of the alphabet continues a word. Keep the complete launcher in the seed.
  JS_PARTS[1] = ';\nconst P = Object.freeze(unpack("';
  if (!JS_PARTS[2].startsWith(']);\n')) throw new Error('Unexpected compiler JavaScript template.');
  JS_PARTS[2] = '"));\n' + `function unpack(text) {
  const words = [];
  let word = 0, place = 1;
  for (const ch of text) {
    let digit = ch.charCodeAt(0) - 48;
    if (digit > 43) digit--;
    word += (digit & 31) * place;
    if (digit < 32) { words.push(word); word = 0; place = 1; }
    else place *= 32;
  }
  return words;
}
` + JS_PARTS[2].slice(4);
  JS_PARTS[3] = `
  return Uint8Array.from(output);
}
function execute(input=P,target=0) {
  const bytes=executeBytes(input,target);
  return target===2?bytes:new TextDecoder("utf-8",{fatal:true}).decode(bytes);
}
` + compilerHost + readText(path.join(root, 'lib/wasm.js')) + printJs + `
const targetNumber=target=>["js","wgsl","wasm"].indexOf(target);
const api={id:"compiler",program:P,execute,executeBytes,
  self:()=>execute(P,0),wgsl:()=>execute(P,1),wasm:()=>execute(P,2),
  compile:(program,target="js")=>execute(program,targetNumber(target)),
  fromWasm:instantiateSprout,
  run:(target="js",input=null,shader=null)=>{
    const selected=targetNumber(target);
    if(selected<0)throw new Error("Use js, wgsl, or wasm.");
    if(input!==null)validateProgram(input);
    return withDevice(device=>dispatchText(device,shader??execute(P,1),"main",
      {TARGET:selected,EXTERNAL:input===null?0:1},input??[],selected===2));
  }
};
const DEFAULT_GPU=false;
` + autorun.replace('const task=', 'const task=args.includes("--wasm")?Promise.resolve({text:api.wasm()}):') + '\n})();\n';

  let binaryRuntime = replaceOnce(/c\s*>\s*127u/, 'c > 255u', WG_RUNTIME);
  binaryRuntime = replaceOnce(/TARGET\s*>\s*1u/, 'TARGET > 2u', binaryRuntime);
  WG_PARTS[2] = WG_PARTS[2].split(WG_RUNTIME).join(binaryRuntime);

  if (!(WASM_OP.length === OPS.length && OPS.length === 19)) throw new Error('Opcode tables disagree.');
  const WASM_LENGTHS = WASM_OP.map((template) => expandWasm(template, [0, 0, 0], 0).length);
  if (WASM_LENGTHS.some((length) => length > 255)) throw new Error('Wasm template is too long.');
  const ALL_TABLES = [...JS_PARTS, ...WG_PARTS, ...JS_OP, ...WG_OP, ...WASM_PARTS, ...WASM_OP, WASM_LENGTHS];
  const data = new Array(2 * ALL_TABLES.length).fill(0);
  ALL_TABLES.forEach((template, index) => {
    const value = typeof template === 'string' ? asciiBytes(template) : template;
    data[index * 2] = 3 + data.length;
    data[index * 2 + 1] = value.length;
    for (const byte of value) data.push(byte);
  });

  const code = [];
  const listing = [];
  const ins = (op, a = 0, b = 0, c = 0) => {
    code.push(OPNUM[op], a, b, c);
    listing.push(`${op} ${a} ${b} ${c}`);
  };
  const note = (text) => listing.push('; ' + text);
  const emitPiece = (k) => {
    ins('set', 5, 2 * k);
    ins('add', 5, 4, 5);
    ins('data', 6, 5);
    ins('add', 5, 5, 3);
    ins('data', 7, 5);
    ins('set', 8, 0);
    ins('lt', 9, 8, 7);
    ins('while', 9);
    ins('add', 11, 6, 8);
    ins('data', 10, 11);
    ins('put', 10);
    ins('add', 8, 8, 3);
    ins('lt', 9, 8, 7);
    ins('end');
  };
  const inputStart = () => {
    ins('input', 13, 3);
    ins('set', 11, 3);
    ins('add', 13, 13, 11);
  };
  const nextCharacter = () => {
    ins('add', 11, 6, 8);
    ins('data', 10, 11);
    ins('add', 8, 8, 3);
  };
  const instructionStart = () => {
    [14, 15, 16, 17].forEach((destination, offset) => {
      ins('set', 11, offset);
      ins('add', 11, 13, 11);
      ins('input', destination, 11);
    });
    ins('mul', 5, 14, 21);
    ins('add', 5, 5, 12);
    ins('data', 6, 5);
    ins('add', 5, 5, 3);
    ins('data', 7, 5);
    ins('set', 8, 0);
    ins('lt', 9, 8, 7);
    ins('while', 9);
    nextCharacter();
  };
  const instructionEnd = () => {
    ins('lt', 9, 8, 7);
    ins('end');
    ins('add', 13, 13, 22);
    ins('lt', 18, 13, 1);
    ins('end');
  };
  const emitLeb = (source, signed = false) => {
    ins('mov', 24, source);
    ins('set', 29, 4);
    ins('while', 29);
    ins('mod', 25, 24, 26);
    ins('add', 25, 25, 26);
    ins('put', 25);
    ins('div', 24, 24, 26);
    ins('sub', 29, 29, 3);
    ins('end');
    if (signed) {
      ins('lt', 30, 24, 31);
      ins('eq', 30, 30, 2);
      ins('if', 30);
      ins('add', 24, 24, 28);
      ins('fi');
    }
    ins('put', 24);
  };
  const literal = (values) => {
    for (const value of values) {
      ins('set', 10, value);
      ins('put', 10);
    }
  };

  note('The same 19-opcode compiler emits direct JS, WGSL, or Wasm.');
  for (const [register, value] of [[2, 0], [3, 1], [21, 2], [22, 4]]) ins('set', register, value);
  ins('lt', 19, 0, 21);
  ins('if', 19);
  note('Text backends share decimal quotation and operand substitution.');
  ins('set', 20, 126);
  ins('set', 23, 8);
  ins('mul', 4, 0, 23);
  ins('set', 11, 3);
  ins('add', 4, 4, 11);
  ins('set', 12, 2 * OPS.length);
  ins('mul', 12, 12, 0);
  ins('set', 11, 19);
  ins('add', 12, 12, 11);
  emitPiece(0);
  ins('decimal', 1);
  emitPiece(1);
  ins('set', 50, 32);
  ins('set', 51, 44);
  ins('set', 52, 48);
  ins('set', 8, 0);
  ins('lt', 9, 8, 1);
  ins('while', 9);
  ins('if', 0);
  ins('if', 8);
  literal([44]);
  ins('fi');
  ins('input', 10, 8);
  ins('decimal', 10);
  ins('else');
  ins('input', 47, 8);
  ins('set', 48, 1);
  ins('while', 48);
  ins('mod', 49, 47, 50);
  ins('div', 47, 47, 50);
  ins('if', 47);
  ins('add', 49, 49, 50);
  ins('fi');
  ins('lt', 48, 49, 51);
  ins('eq', 48, 48, 2);
  ins('if', 48);
  ins('add', 49, 49, 3);
  ins('fi');
  ins('add', 49, 49, 52);
  ins('put', 49);
  ins('mov', 48, 47);
  ins('end');
  ins('fi');
  ins('add', 8, 8, 3);
  ins('lt', 9, 8, 1);
  ins('end');
  emitPiece(2);
  inputStart();
  ins('lt', 18, 13, 1);
  ins('while', 18);
  instructionStart();
  ins('eq', 19, 10, 20);
  ins('if', 19);
  nextCharacter();
  ins('mov', 32, 17);
  for (const [character, register] of [[97, 15], [98, 16]]) {
    ins('set', 11, character);
    ins('eq', 19, 10, 11);
    ins('if', 19);
    ins('mov', 32, register);
    ins('fi');
  }
  ins('decimal', 32);
  ins('else');
  ins('put', 10);
  ins('fi');
  instructionEnd();
  emitPiece(3);
  ins('else');
  note('Native Wasm: count fixed template lengths, emit code, quote input words.');
  for (const [register, value] of [[4, 3 + 46 * 2], [12, 3 + 51 * 2], [20, 255],
    [26, 128], [28, 112], [31, 8], [33, 256], [39, 3]]) ins('set', register, value);
  ins('set', 11, 3 + 70 * 2);
  ins('data', 37, 11);
  ins('set', 34, WASM_PARTS[2].length + 5 + WASM_PARTS[3].length + WASM_PARTS[4].length);
  inputStart();
  ins('lt', 18, 13, 1);
  ins('while', 18);
  ins('input', 14, 13);
  ins('add', 11, 37, 14);
  ins('data', 38, 11);
  ins('add', 34, 34, 38);
  ins('add', 13, 13, 22);
  ins('lt', 18, 13, 1);
  ins('end');
  ins('set', 35, WASM_PARTS[1].length + 5);
  ins('add', 35, 35, 34);
  emitPiece(0);
  emitLeb(35);
  emitPiece(1);
  emitLeb(34);
  emitPiece(2);
  emitLeb(1, true);
  emitPiece(3);
  inputStart();
  ins('lt', 18, 13, 1);
  ins('while', 18);
  instructionStart();
  ins('eq', 19, 10, 20);
  ins('if', 19);
  nextCharacter();
  ins('set', 11, 105);
  ins('eq', 19, 10, 11);
  ins('if', 19);
  emitLeb(16, true);
  ins('else');
  ins('set', 11, 110);
  ins('eq', 19, 10, 11);
  ins('if', 19);
  emitLeb(1, true);
  ins('else');
  ins('mov', 32, 17);
  for (const [character, register] of [[97, 15], [98, 16]]) {
    ins('set', 11, character);
    ins('eq', 19, 10, 11);
    ins('if', 19);
    ins('mov', 32, register);
    ins('fi');
  }
  ins('add', 32, 32, 39);
  ins('put', 32);
  ins('fi');
  ins('fi');
  ins('else');
  ins('put', 10);
  ins('fi');
  instructionEnd();
  emitPiece(4);
  literal([11]);
  ins('mul', 35, 1, 22);
  ins('set', 11, 10);
  ins('add', 34, 35, 11);
  emitLeb(34);
  literal([1, 0, 65, 0, 11]);
  emitLeb(35);
  ins('set', 13, 0);
  ins('lt', 18, 13, 1);
  ins('while', 18);
  ins('input', 44, 13);
  ins('set', 46, 4);
  ins('while', 46);
  ins('mod', 45, 44, 33);
  ins('put', 45);
  ins('div', 44, 44, 33);
  ins('sub', 46, 46, 3);
  ins('end');
  ins('add', 13, 13, 3);
  ins('lt', 18, 13, 1);
  ins('end');
  ins('fi');

  const PROGRAM = [21331, data.length, code.length, ...data, ...code];
  if (PROGRAM.length > 60000) throw new Error('The compiler program exceeds 60000 words.');

  function bootstrapCompile(p, target) {
    const instructions = [];
    for (let i = 3 + p[1]; i < p.length; i += 4) instructions.push(p.slice(i, i + 4));
    if (target === 2) {
      let body = bytes(WASM_PARTS[2], padded(p.length, true), WASM_PARTS[3]);
      for (const [op, a, b, c] of instructions) body = body.concat(expandWasm(WASM_OP[op], [a, b, c], p.length));
      body = body.concat(WASM_PARTS[4]);
      const codePayload = bytes(WASM_PARTS[1], padded(body.length), body);
      const quoted = littleEndianWords(p);
      const dataPayload = bytes(0x01, 0x00, 0x41, 0x00, 0x0b, padded(quoted.length), quoted);
      return Uint8Array.from(bytes(WASM_PARTS[0], padded(codePayload.length), codePayload, 0x0b, padded(dataPayload.length), dataPayload));
    }
    const [parts, templates] = target === 0 ? [JS_PARTS, JS_OP] : [WG_PARTS, WG_OP];
    const body = instructions.map(([op, a, b, c]) => templates[op].replaceAll('~a', String(a)).replaceAll('~b', String(b)).replaceAll('~c', String(c))).join('');
    const quoted = target === 0 ? packedWords(p) : numbers(p);
    return parts[0] + String(p.length) + parts[1] + quoted + parts[2] + body + parts[3];
  }

  function writeProgram(stem, p, assembly = null) {
    ['js', 'wgsl', 'wasm'].forEach((extension, target) => {
      const content = bootstrapCompile(p, target);
      fs.writeFileSync(path.join(root, `${stem}.${extension}`), typeof content === 'string' ? asciiBytes(content) : content);
    });
    if (assembly !== null) writeText(path.join(root, `${stem}.sprout`), assembly);
  }

  writeProgram('quines/05-compiler', PROGRAM);
  writeText(path.join(root, 'compiler/compiler-program.json'), jsonText(PROGRAM) + '\n');
  writeText(path.join(root, 'compiler/compiler.sprout'), assemblyText(PROGRAM, listing));

  const sample = (rows, values = []) => {
    const words = [];
    const lines = [];
    for (const [op, ...given] of rows) {
      const args = [...given, ...new Array(3 - given.length).fill(0)];
      words.push(OPNUM[op], ...args);
      lines.push(op + ' ' + args.join(' '));
    }
    const p = [21331, values.length, words.length, ...values, ...words];
    return [p, assemblyText(p, lines)];
  };

  const exampleRecords = [];
  for (const [name, [expected, rows]] of Object.entries(EXAMPLES)) {
    const [p, assembly] = sample(rows, name === 'hello' ? [...asciiBytes(expected)] : []);
    writeProgram('compiler/' + name, p, assembly);
    writeText(path.join(root, `compiler/${name}.json`), jsonText(p) + '\n');
    exampleRecords.push({ name, expected });
  }
  writeText(path.join(root, 'compiler/examples.json'), jsonText(exampleRecords, { indent: 2 }) + '\n');

  const measurements = {
    schema: 'sprout-size-v1', opcodes: OPS.length, registers: 64,
    instructions: Math.floor(code.length / 4), dataWords: data.length, programWords: PROGRAM.length,
    measurement: 'Byte counts and deterministic gzip, not Kolmogorov complexity.',
    encoding: 'Five-byte LEB128 for dynamic sizes and i32 constants; direct native instructions.',
    jsQuotation: 'Printable five-bit word encoding; full compiler and launcher remain in the seed.',
    files: {},
  };
  for (const relative of ['compiler/compiler.sprout', 'compiler/compiler-program.json',
    'quines/05-compiler.js', 'quines/05-compiler.wgsl', 'quines/05-compiler.wasm']) {
    const content = fs.readFileSync(path.join(root, relative));
    measurements.files[relative] = {
      bytes: content.length, gzipBytes: gzipLevel9(content).length,
      sha256: createHash('sha256').update(content).digest('hex'),
    };
  }
  writeText(path.join(root, 'compiler/complexity.json'), jsonText(measurements, { indent: 2 }) + '\n');
  console.log(`Sprout compiler: ${PROGRAM.length} words; ${Math.floor(code.length / 4)} instructions; ${data.length} data words; 3 backends`);
  return { listing };
}
