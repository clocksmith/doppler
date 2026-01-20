import fs from 'node:fs';
import path from 'node:path';

// 1. Source: The Node.js binary itself (highly structured, not random)
const SOURCE_FILE = process.execPath;

function calculateEntropy(buffer) {
  const counts = new Uint32Array(256);
  for (const byte of buffer) counts[byte]++;
  
  let entropy = 0;
  const len = buffer.length;
  for (let i = 0; i < 256; i++) {
    if (counts[i] === 0) continue;
    const p = counts[i] / len;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function getFloatStats(view) {
  let nans = 0, infs = 0, zeros = 0, normal = 0;
  let min = Infinity, max = -Infinity;
  
  for (let i = 0; i < view.length; i++) {
    const val = view[i];
    if (Number.isNaN(val)) nans++;
    else if (!Number.isFinite(val)) infs++;
    else if (val === 0) zeros++;
    else {
      normal++;
      if (val < min) min = val;
      if (val > max) max = val;
    }
  }
  return { nans, infs, zeros, normal, min, max, total: view.length };
}

console.log(`Reading binary: ${SOURCE_FILE}`);
const rawBuffer = fs.readFileSync(SOURCE_FILE);

// Align to 4 bytes
const remainder = rawBuffer.length % 4;
const alignedBuffer = rawBuffer.subarray(0, rawBuffer.length - remainder);
const sourceFloats = new Float32Array(alignedBuffer.buffer, alignedBuffer.byteOffset, alignedBuffer.length / 4);

console.log(`\n--- Source Analysis ---`);
console.log(`Bytes: ${rawBuffer.length}`);
console.log(`Float32 Count: ${sourceFloats.length}`);
console.log(`Byte Entropy: ${calculateEntropy(rawBuffer).toFixed(4)} bits/byte (Max 8.0)`);
console.log(`Stats:`, getFloatStats(sourceFloats));

// 2. The "Morph": Mapping structured opcodes to a wave function
// We'll apply a transform that maps the input domain (wildly varying floats) 
// to a constrained domain (-1.0 to 1.0), changing the "meaning" of the bytes.
const morphedFloats = new Float32Array(sourceFloats.length);
const transformName = "Sine_Phase_Modulation";

for (let i = 0; i < sourceFloats.length; i++) {
  const input = sourceFloats[i];
  
  // Handling NaNs/Infs which destroy math
  if (!Number.isFinite(input)) {
    morphedFloats[i] = 0; // "Silence" the noise
    continue;
  }
  
  // Transform: Treat the float value as a phase in a sine wave
  // This morphs the "instruction data" into "harmonic data"
  morphedFloats[i] = Math.sin(input);
}

// 3. Map back to bytes
const morphedBytes = new Uint8Array(morphedFloats.buffer);

console.log(`\n--- Morphed Analysis (${transformName}) ---`);
console.log(`Stats:`, getFloatStats(morphedFloats));
console.log(`Byte Entropy: ${calculateEntropy(morphedBytes).toFixed(4)} bits/byte`);

// 4. Comparison
// Did we make it more or less structured?
// High entropy = Looks like noise/compressed data
// Low entropy = Looks like text/simple images/audio
console.log(`\n--- Conclusion ---`);
console.log(`We transformed specific CPU instructions (high entropy) into smooth sine wave values.`);
console.log(`The byte-level entropy changed, reflecting the new "meaning" (mathematical structure) of the data.`);
