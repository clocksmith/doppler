import { quantizeToQ4KM, QK4_K_BLOCK_SIZE, QK_K, float16ToFloat32 } from '../src/converter/quantizer.js';

// Create test data with known pattern
const testData = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  testData[i] = Math.sin(i * 0.1) * 10;  // Values in range [-10, 10]
}

// Quantize
const result = quantizeToQ4KM(testData, [256]);
console.log('Quantized size:', result.quantizedSize);

// Manual dequant to check
const block = result.quantized;
const view = new DataView(block.buffer);
const d = float16ToFloat32(view.getUint16(0, true));
const dmin = float16ToFloat32(view.getUint16(2, true));

console.log('d:', d.toFixed(6), 'dmin:', dmin.toFixed(6));

// Extract scales/mins
/** @type {number[]} */
const scaleBits = [];
/** @type {number[]} */
const minBits = [];
for (let i = 0; i < 4; i++) {
  scaleBits[i] = block[4 + i] & 0x3f;
  scaleBits[i + 4] = ((block[4 + i] >> 6) & 0x03) << 4;
}
for (let i = 0; i < 4; i++) {
  minBits[i] = block[4 + 4 + i] & 0x3f;
  minBits[i + 4] = ((block[4 + 4 + i] >> 6) & 0x03) << 4;
}
for (let i = 0; i < 4; i++) {
  scaleBits[i + 4] |= block[4 + 8 + i] & 0x0f;
  minBits[i + 4] |= (block[4 + 8 + i] >> 4) & 0x0f;
}

const scales = scaleBits.map(s => d * s);
const mins = minBits.map(m => dmin * m);

console.log('\nDequant test (CPU reference):');
// Check values across subblocks
for (const i of [0, 31, 32, 63, 64, 127, 128, 191, 192, 255]) {
  const chunk = Math.floor(i / 64);
  const pos = i % 64;
  const useUpper = pos >= 32;
  const byteInRange = useUpper ? pos - 32 : pos;
  const byteIdx = 16 + chunk * 32 + byteInRange;
  const byte = block[byteIdx];
  const q = useUpper ? (byte >> 4) & 0xf : byte & 0xf;
  const sb = Math.floor(i / 32);
  const dequant = scales[sb] * q - mins[sb];
  const error = Math.abs(testData[i] - dequant);
  console.log(`[${i.toString().padStart(3)}] orig=${testData[i].toFixed(3).padStart(7)} dequant=${dequant.toFixed(3).padStart(7)} err=${error.toFixed(3)} sb=${sb}`);
}

// Compute MSE
let mse = 0;
for (let i = 0; i < 256; i++) {
  const chunk = Math.floor(i / 64);
  const pos = i % 64;
  const useUpper = pos >= 32;
  const byteInRange = useUpper ? pos - 32 : pos;
  const byteIdx = 16 + chunk * 32 + byteInRange;
  const byte = block[byteIdx];
  const q = useUpper ? (byte >> 4) & 0xf : byte & 0xf;
  const sb = Math.floor(i / 32);
  const dequant = scales[sb] * q - mins[sb];
  mse += (testData[i] - dequant) ** 2;
}
mse /= 256;
console.log('\nMSE:', mse.toFixed(6));
console.log('RMSE:', Math.sqrt(mse).toFixed(6));
