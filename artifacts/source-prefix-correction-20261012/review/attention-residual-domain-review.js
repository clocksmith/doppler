// Independent CPU domain review; never imported by runtime or an acceptance reference.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const scratch = new DataView(new ArrayBuffer(4));
const fromBits = bits => { scratch.setUint32(0, bits); return scratch.getFloat32(0); };
const toBits = value => { scratch.setFloat32(0, value); return scratch.getUint32(0); };
const hex = value => `0x${value.toString(16).padStart(8, '0')}`;
const helperPath = new URL('../candidate-range-product-rmsnorm.wgsl', import.meta.url);
const helperSha256 = hash(fs.readFileSync(helperPath));
assert.equal(helperSha256, 'a056cdd5a773d12d624a974d4c48bf57f9504bf4713496e44c5dfcf374fe9fc5');

// Translate the existing addition helper's low-word materialization exactly for
// this finite counterexample. BigInt is a reviewer instrument, never runtime math.
function additionResidual(aBits, bBits, sumBits) {
  const a = fromBits(aBits), b = fromBits(bBits), sum = fromBits(sumBits);
  if (sum === a) return bBits;
  if (sum === b) return aBits;
  const exponent = Math.max((aBits >>> 23) & 255, (bBits >>> 23) & 255, (sumBits >>> 23) & 255, 1);
  if (exponent === 255) return 0;
  const align = bits => {
    const stored = (bits >>> 23) & 255;
    const mantissa = BigInt((bits & 0x7fffff) | (stored ? 0x800000 : 0));
    const magnitude = (mantissa << 32n) >> BigInt(exponent - Math.max(stored, 1));
    return bits >>> 31 ? -magnitude : magnitude;
  };
  const residual = BigInt.asIntN(64, align(aBits) + align(bBits) - align(sumBits));
  const magnitude = residual < 0n ? -residual : residual;
  const lowWord = Number(magnitude & 0xffffffffn);
  return toBits(Math.fround(lowWord * 2 ** (exponent - 182) * (residual < 0n ? -1 : 1)));
}
const counterexamples = [];
for (const sign of [0, 0x80000000]) {
  const a = (0x00800000 ^ sign) >>> 0;
  const b = (0x807fffff ^ sign) >>> 0;
  const sum = sign;
  const expected = (1 ^ sign) >>> 0;
  const observed = additionResidual(a, b, sum);
  assert.notEqual(observed, expected);
  counterexamples.push({ a: hex(a), b: hex(b), suppliedSum: hex(sum),
    expectedResidual: hex(expected), translatedHelperResidual: hex(observed),
    condition: 'Finite cancellation whose exact subnormal sum was flushed to signed zero. Not an observed GPU result.' });
}
// Ordinary nearest-even non-FTZ addition provides a control for the same inputs.
assert.equal(additionResidual(0x00800000, 0x807fffff, 1), 0);

const capturePath = process.argv[2];
assert.ok(capturePath, 'Supply the retained decode7 capture path');
const bytes = fs.readFileSync(capturePath);
const captureSha256 = hash(bytes);
assert.equal(captureSha256, '4d377a97fd0f1d4aa3f9a4e92900a35fa3b8b33bca0846e72c1e9e85dc731d18');
const capture = JSON.parse(bytes);
const q = capture.observation.timeline.find(row => row.layerIndex === 15 && row.stageName === 'attn.q_rope');
const kv = capture.observation.physicalCache.find(row => row.layer === 15 && row.step === 0);
assert.deepEqual(q.capture.shape, [184, 2048]);
assert.deepEqual(kv.shape, [184, 2, 256]);
assert.equal(q.capture.encoding, 'base64-f32le');
assert.equal(kv.values.k.encoding, 'base64-f16le');
function scan(bytes, half) {
  let nonfinite = 0, zero = 0, subnormal = 0, minimum = Infinity, maximum = 0, minimumQuantumPower = Infinity;
  const width = half ? 2 : 4;
  for (let i = 0; i < bytes.length; i += width) {
    const bits = half ? bytes.readUInt16LE(i) : bytes.readUInt32LE(i);
    const exponent = half ? (bits >>> 10) & 31 : (bits >>> 23) & 255;
    const mantissa = bits & (half ? 1023 : 0x7fffff);
    if (exponent === (half ? 31 : 255)) { nonfinite++; continue; }
    if (exponent === 0 && mantissa === 0) { zero++; continue; }
    if (exponent === 0) subnormal++;
    const quantumPower = half ? Math.max(exponent, 1) - 25 : Math.max(exponent, 1) - 150;
    const value = half ? (mantissa + (exponent ? 1024 : 0)) * 2 ** quantumPower : Math.abs(fromBits(bits));
    minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
    minimumQuantumPower = Math.min(minimumQuantumPower, quantumPower);
  }
  return { count: bytes.length / width, nonfinite, zero, subnormal, minimum, maximum, minimumQuantumPower };
}
const qBytes = Buffer.from(q.capture.dataBase64, 'base64');
const kBytes = Buffer.from(kv.values.k.dataBase64, 'base64');
const qRange = scan(qBytes, false), kRange = scan(kBytes, true);
const productQuantumPower = qRange.minimumQuantumPower + kRange.minimumQuantumPower;
// Very conservative magnitude enclosure for 256 products and their low parts.
const partialMagnitudeBound = 512 * qRange.maximum * kRange.maximum;
const finiteNormalLattice = qRange.nonfinite === 0 && kRange.nonfinite === 0
  && productQuantumPower >= -126 && partialMagnitudeBound < 2 ** 127;
assert.equal(finiteNormalLattice, true);
console.log(JSON.stringify({ schema: 'doppler-attention-residual-domain-review-v1',
  checkerSha256: hash(fs.readFileSync(new URL(import.meta.url))), helperSha256,
  capturePath, captureSha256, qBytesSha256: hash(qBytes), kBytesSha256: hash(kBytes),
  layer: 15, phase: 'prefill', queries: 184, heads: 8, kvHeads: 2, headDim: 256,
  qRange, kRange, productQuantumPower, partialMagnitudeBound, finiteNormalLattice,
  counterexamples,
  scope: 'Captured Q/F16 K lattice and magnitude bound plus CPU helper counterexample. No new WGSL candidate, no GPU execution, no causal or whole-model accuracy qualification.',
  conditions: [
    'Captured F16 K is promoted exactly to F32; nonzero products and finite nearest-even accumulation residuals remain multiples of the stated binary quantum.',
    'The bound covers this captured score dot-product domain, not arbitrary application Q/K or actual backend reassociation/rounding modes.',
    'A generic helper claim must cover FTZ/cancellation or enforce a valid domain and retain baseline behavior outside it.',
    'Product recovery alone does not make correction accumulation, score scaling, exponential, softmax, or P*V accumulation exact.'
  ]
}, null, 2));
