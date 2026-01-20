import fs from 'node:fs';
import { execSync } from 'node:child_process';
import path from 'node:path';

const SHARD_PATH = 'models/google-gemma-3-1b-it-wf16/shard_00008.bin';
const EXP_DIR = 'experimental';
const RAW_FILE = path.join(EXP_DIR, 'raw.bin');
const MORPHED_FILE = path.join(EXP_DIR, 'morphed.bin');

console.log(`Reading shard: ${SHARD_PATH}`);
const rawBuffer = fs.readFileSync(SHARD_PATH);

// 1. Write Raw
console.log(`Writing raw file to ${RAW_FILE}...`);
fs.writeFileSync(RAW_FILE, rawBuffer);

// 2. Generate Morphed (Tanh transform)
console.log(`Generating morphed data (Tanh transform)...`);
// Ensure alignment
const floatCount = Math.floor(rawBuffer.length / 4);
const rawFloats = new Float32Array(rawBuffer.buffer, rawBuffer.byteOffset, floatCount);
const morphedFloats = new Float32Array(floatCount);

for (let i = 0; i < floatCount; i++) {
  const val = rawFloats[i];
  // Tanh morph
  if (Number.isFinite(val)) {
    morphedFloats[i] = Math.tanh(val) * 100;
  } else {
    morphedFloats[i] = 0;
  }
}

const morphedBuffer = Buffer.from(morphedFloats.buffer);
console.log(`Writing morphed file to ${MORPHED_FILE}...`);
fs.writeFileSync(MORPHED_FILE, morphedBuffer);

// 3. Compress
console.log('Compressing files with gzip...');
try {
  execSync(`gzip -k -f ${RAW_FILE}`);
  execSync(`gzip -k -f ${MORPHED_FILE}`);
} catch (e) {
  console.error("Compression failed:", e.message);
}

// 4. Compare
const rawSize = fs.statSync(RAW_FILE).size;
const rawGzSize = fs.statSync(`${RAW_FILE}.gz`).size;
const morphedSize = fs.statSync(MORPHED_FILE).size;
const morphedGzSize = fs.statSync(`${MORPHED_FILE}.gz`).size;

console.log('\n--- Compression Results ---');
console.log(`Raw File:     ${(rawSize / 1024 / 1024).toFixed(2)} MB`);
console.log(`Raw GZIP:     ${(rawGzSize / 1024 / 1024).toFixed(2)} MB  (Ratio: ${(rawSize / rawGzSize).toFixed(2)}x)`);
console.log(`---------------------------`);
console.log(`Morphed File: ${(morphedSize / 1024 / 1024).toFixed(2)} MB`);
console.log(`Morphed GZIP: ${(morphedGzSize / 1024 / 1024).toFixed(2)} MB  (Ratio: ${(morphedSize / morphedGzSize).toFixed(2)}x)`);

if (rawGzSize < morphedGzSize) {
  console.log(`\nCONCLUSION: The RAW binary is more compressible.`);
} else {
  console.log(`\nCONCLUSION: The MORPHED binary is more compressible.`);
}
