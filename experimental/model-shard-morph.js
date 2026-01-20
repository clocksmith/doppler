import fs from 'node:fs';
import path from 'node:path';

const INPUT_SHARD = 'models/google-gemma-3-1b-it-wf16/shard_00000.bin';

function calculateStats(buffer) {
  let printable = 0;
  const counts = new Uint32Array(256);
  for (const byte of buffer) {
    counts[byte]++;
    // ASCII Printable range: 32 (space) to 126 (~)
    if (byte >= 32 && byte <= 126) printable++;
  }
  
  let entropy = 0;
  for (let i = 0; i < 256; i++) {
    if (counts[i] === 0) continue;
    const p = counts[i] / buffer.length;
    entropy -= p * Math.log2(p);
  }
  
  return {
    entropy: entropy.toFixed(4),
    printablePercent: ((printable / buffer.length) * 100).toFixed(2) + '%'
  };
}

console.log(`Analyzing Model Shard: ${INPUT_SHARD}`);
const rawBuffer = fs.readFileSync(INPUT_SHARD);
const originalStats = calculateStats(rawBuffer);

console.log(`\n--- Original "Noisy" Weight Data ---`);
console.log(`Entropy: ${originalStats.entropy} bits/byte`);
console.log(`Printable ASCII Bytes: ${originalStats.printablePercent}`);

// Transformation: Tanh Morph
// This treats the weights as inputs to a neuron and squashes them.
const floatView = new Float32Array(
  rawBuffer.buffer, 
  rawBuffer.byteOffset, 
  Math.floor(rawBuffer.length / 4)
);

const morphedBuffer = Buffer.alloc(rawBuffer.length);
const morphedFloats = new Float32Array(
  morphedBuffer.buffer, 
  morphedBuffer.byteOffset, 
  floatView.length
);

for (let i = 0; i < floatView.length; i++) {
  const x = floatView[i];
  if (!Number.isFinite(x)) {
    morphedFloats[i] = 0;
    continue;
  }
  // Tanh(x) maps the infinite float space to [-1, 1]
  // We then scale it by 100 to shift the "byte meaning" into the printable range
  morphedFloats[i] = Math.tanh(x) * 100;
}

const morphedStats = calculateStats(morphedBuffer);

console.log(`\n--- Morphed "Neural" Data (Tanh * 100) ---`);
console.log(`Entropy: ${morphedStats.entropy} bits/byte`);
console.log(`Printable ASCII Bytes: ${morphedStats.printablePercent}`);

console.log(`\n--- Result ---`);
console.log(`The weights (which look like noise) were squashed into a specific mathematical range.`);
console.log(`Notice the printable ASCII percentage changed. We are shifting the "meaning" of the bits`);
console.log(`from abstract floating point numbers toward something that occupies a different part of the byte-space.`);

// Extra: Peek at the first 64 bytes of "text" we created from the weights
console.log(`\nRaw Peek (First 64 morphed bytes as string):`);
console.log(morphedBuffer.subarray(0, 64).toString('ascii').replace(/[^ -~]/g, '.'));
