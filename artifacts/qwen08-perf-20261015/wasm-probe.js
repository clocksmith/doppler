import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { f16ToF32 } from '../../src/loader/dtype-utils.js';
const bytes = await fs.readFile(new URL('./half-convert.wasm', import.meta.url));
const began = performance.now();
const { instance } = await WebAssembly.instantiate(bytes);
const startupMs = performance.now() - began;
const { memory, convert } = instance.exports;
const inputOffset = 1024 * 1024;
const outputOffset = 32 * 1024 * 1024;
const input = new Uint16Array(memory.buffer, inputOffset, 65536);
for (let i = 0; i < input.length; i++) input[i] = i;
convert(inputOffset, outputOffset, input.length);
const output = new Float32Array(memory.buffer, outputOffset, input.length);
for (let i = 0; i < input.length; i++) assert.ok(Object.is(output[i], f16ToF32(i)), `half bits ${i}`);
convert(inputOffset, outputOffset, 65535);
for (let i = 65532; i < 65535; i++) assert.ok(Object.is(output[i], f16ToF32(i)));
function convertBits(source) {
  const result = new Float32Array(source.length);
  const bits = new Uint32Array(result.buffer);
  for (let i = 0; i < source.length; i++) {
    const h = source[i], exponent = (h >>> 10) & 31, mantissa = h & 1023;
    if (exponent === 0) { result[i] = (h & 32768 ? -1 : 1) * mantissa * 2 ** -24; }
    else { bits[i] = ((h & 32768) << 16) | (exponent === 31 ? 0x7f800000 : (exponent + 112) << 23) | (mantissa << 13); }
  }
  return result;
}
const optimized = convertBits(input);
for (let i = 0; i < input.length; i++) assert.ok(Object.is(optimized[i], f16ToF32(i)));
const rows = [];
for (const count of [65536, 4194304]) {
  const source = Uint16Array.from({length: count}, (_, i) => i & 65535);
  const wasmInput = new Uint16Array(memory.buffer, inputOffset, count);
  const wasmOutput = new Float32Array(memory.buffer, outputOffset, count);
  const times = { js: [], jsBitConversion: [], wasmWithCopies: [], wasmResident: [] };
  let consumed = 0;
  for (let run = 0; run < 7; run++) {
    const tasks = {
      jsBitConversion() { return convertBits(source); },
      js() { const out = new Float32Array(count); for (let i=0;i<count;i++) out[i]=f16ToF32(source[i]); return out; },
      wasmWithCopies() { wasmInput.set(source); convert(inputOffset, outputOffset, count); return wasmOutput.slice(); },
      wasmResident() { convert(inputOffset, outputOffset, count); return wasmOutput; },
    };
    for (const name of run % 2 ? Object.keys(tasks).reverse() : Object.keys(tasks)) {
      const start = performance.now(); const values = tasks[name](); const elapsed = performance.now()-start;
      consumed += values[1]; if (run > 1) times[name].push(elapsed);
    }
  }
  rows.push({count, inputBytes:source.byteLength, outputBytes:count*4, times, consumed});
}
const result = {scope:'CPU F16 artifact materialization only; not inference', startupMs,
  wasmSha256:createHash('sha256').update(bytes).digest('hex'), checkedHalfPatterns:65536,
  tailChecked:true, wasmMemoryBytes:memory.buffer.byteLength, node:process.version, rows};
await fs.writeFile(new URL('./wasm-probe.json', import.meta.url), JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
