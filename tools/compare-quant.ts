// Compare converter quantization vs reference quantization
import { quantizeToQ4KM } from '../src/converter/quantizer.js';
import { quantizeQ4_KRef, dequantQ4_KRef } from '../kernel-tests/src/reference/dequant.js';

const numBlocks = 2;
const blockSize = 256;
const total = numBlocks * blockSize;

// Create deterministic test data
const values = new Float32Array(total);
for (let i = 0; i < total; i++) {
  values[i] = Math.sin(i * 0.1) * 0.75 + Math.cos(i * 0.03) * 0.25;
}

console.log('Input values (first 10):', Array.from(values.slice(0, 10)).map(x => x.toFixed(4)));

// Quantize with both
const converterResult = quantizeToQ4KM(values, [total]);
const refResult = quantizeQ4_KRef(values, numBlocks);

console.log('Converter quantized size:', converterResult.quantized.length);
console.log('Reference quantized size:', refResult.length);

// Compare bytes
let byteDiffs = 0;
for (let i = 0; i < converterResult.quantized.length; i++) {
  if (converterResult.quantized[i] !== refResult[i]) {
    if (byteDiffs < 10) {
      console.log(`Byte diff at ${i}: converter=${converterResult.quantized[i]}, ref=${refResult[i]}`);
    }
    byteDiffs++;
  }
}
console.log('Total byte differences:', byteDiffs);

// Dequantize with reference
const converterDequant = dequantQ4_KRef(converterResult.quantized, numBlocks);
const refDequant = dequantQ4_KRef(refResult, numBlocks);

// Compare dequantized values
let maxDiff = 0;
let maxDiffIdx = -1;
for (let i = 0; i < total; i++) {
  const diff = Math.abs(converterDequant[i] - refDequant[i]);
  if (diff > maxDiff) {
    maxDiff = diff;
    maxDiffIdx = i;
  }
}
console.log('Max dequant diff between converter and ref:', maxDiff, 'at index', maxDiffIdx);

// Compare to original
let converterError = 0;
let refError = 0;
for (let i = 0; i < total; i++) {
  converterError = Math.max(converterError, Math.abs(values[i] - converterDequant[i]));
  refError = Math.max(refError, Math.abs(values[i] - refDequant[i]));
}
console.log('Max reconstruction error - converter:', converterError);
console.log('Max reconstruction error - reference:', refError);

// Print first block's bytes
const converterBlock = converterResult.quantized.slice(0, 144);
const refBlock = refResult.slice(0, 144);

console.log('\nBlock 0 comparison:');
console.log('Converter d+dmin bytes:', converterBlock[0], converterBlock[1], converterBlock[2], converterBlock[3]);
console.log('Reference d+dmin bytes:', refBlock[0], refBlock[1], refBlock[2], refBlock[3]);
console.log('Converter scales (bytes 4-15):', Array.from(converterBlock.slice(4, 16)));
console.log('Reference scales (bytes 4-15):', Array.from(refBlock.slice(4, 16)));
console.log('Converter qs (first 16):', Array.from(converterBlock.slice(16, 32)));
console.log('Reference qs (first 16):', Array.from(refBlock.slice(16, 32)));
