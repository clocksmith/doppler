import test from 'node:test';
import assert from 'node:assert/strict';
import { fma } from './dyadic-diagnostic.js';
import {
  f32ToBits,
  bitsToF32,
  roundTiesToEven,
  fmaExact,
  reciprocalRefined,
  divideRefined,
  inverseRootRefined,
  expRefined,
  siluWgsl,
  sigmoidRefined,
  logRefined,
  softplusWgsl,
  witnessHyperConnectionNorm,
  witnessLowRankSilu,
  witnessMixSigmoid,
  witnessInjectGate,
  witnessHyperConnectionBranch,
  witnessHyperConnectionInject,
  witnessMatmulRowSubgroupWave64,
  witnessMatmulF16wF32aTiled16,
  witnessMatmulF16wF32aSeq,
  witnessGatedDeltaCore,
  compareF32Bits,
  exactDyadicPairMargin,
} from './layer25-wgsl-witness-core.js';

test('roundTiesToEven matches WGSL round semantics on positive and negative half-integers', () => {
  assert.equal(roundTiesToEven(0.5), 0);
  assert.equal(roundTiesToEven(1.5), 2);
  assert.equal(roundTiesToEven(2.5), 2);
  assert.equal(roundTiesToEven(3.5), 4);
  assert.equal(roundTiesToEven(-0.5), 0);
  assert.equal(roundTiesToEven(-1.5), -2);
  assert.equal(roundTiesToEven(-2.5), -2);
  assert.equal(roundTiesToEven(-3.5), -4);
  assert.throws(() => roundTiesToEven(NaN), /Finite input required/);
});

test('fmaExact agrees bit-for-bit with dyadic fma including double-rounding midpoint cases', () => {
  // Construct a double-rounding case:
  // a = 2^-11 * (1 + 2^-23), b = 2^-12 * (1 - 2^-23) => exact product p = a * b = 2^-23 - 2^-69.
  // Choose c = 2 + 2^-22 (an F32 number in [2, 4) with odd LSB = 1, where F32 ULP = 2^-22 and half-ULP = 2^-23).
  // Exact p + c = (2 + 2^-22) + 2^-23 - 2^-69, which is strictly below the midpoint (2 + 2^-22) + 2^-23.
  // True single-rounded FMA must round down to c = 2 + 2^-22.
  // Naive binary64 (a * b + c) rounds 2 + 2^-22 + 2^-23 - 2^-69 up to the exact midpoint 2 + 3 * 2^-23,
  // and Math.fround then rounds ties-to-even up to 2 + 2 * 2^-22!
  const a = (2 ** -11) * (1 + 2 ** -23);
  const b = (2 ** -12) * (1 - 2 ** -23);
  const c = 2 + 2 ** -22;
  const naive = Math.fround(a * b + c);
  const exactDyadic = fma(a, b, c);
  const exactFast = fmaExact(a, b, c);
  assert.equal(naive, 2 + 2 * (2 ** -22));
  assert.equal(exactDyadic, c);
  assert.equal(f32ToBits(exactFast), f32ToBits(exactDyadic));
});

test('WGSL transcendental refinements and activations are finite and self-consistent', () => {
  assert.equal(expRefined(0), 1);
  assert.equal(logRefined(1), 0);
  assert.equal(sigmoidRefined(0), 0.5);
  assert.equal(siluWgsl(0), 0);
  const sp0 = softplusWgsl(0);
  assert.ok(Math.abs(sp0 - Math.fround(Math.LN2)) <= 2 ** -23);
  const d = divideRefined(1, 4);
  assert.equal(d, 0.25);
  const inv = inverseRootRefined(4);
  assert.equal(inv, 0.5);
  assert.equal(reciprocalRefined(2), 0.5);
});

test('witnessHyperConnectionNorm separates left-associated offset witness from negative controls', () => {
  const streamCount = 4;
  const streamWidth = 2560;
  const rows = 1;
  const input = new Float32Array(rows * streamCount * streamWidth);
  const weight = new Float32Array(streamCount * streamWidth);
  for (let i = 0; i < input.length; i++) {
    input[i] = Math.fround(((i % 97) - 48) * 0.03125);
    weight[i] = Math.fround(((i % 53) - 26) * 0.0078125);
  }
  const actual = witnessHyperConnectionNorm(input, weight, rows);
  const match = compareF32Bits(actual, actual);
  assert.equal(match.allExact, true);
  assert.equal(match.maxUlp, 0);

  const wrongOffset = witnessHyperConnectionNorm(input, weight, rows, { weightOffset: false });
  assert.equal(compareF32Bits(wrongOffset, actual).allExact, false);

  const rightAssoc = witnessHyperConnectionNorm(input, weight, rows, { rightAssociated: true });
  assert.equal(compareF32Bits(rightAssoc, actual).allExact, false);

  const corrupted = new Float32Array(actual);
  corrupted[17] = bitsToF32(f32ToBits(corrupted[17]) + 1);
  const corrCmp = compareF32Bits(actual, corrupted);
  assert.equal(corrCmp.allExact, false);
  assert.equal(corrCmp.exact, actual.length - 1);
  assert.equal(corrCmp.maxUlp, 1);
});

test('witnessMatmulRowSubgroupWave64 distinguishes bottom-up DPP tree from top-down reduction', () => {
  const rows = 1;
  const K = 512;
  const N = 8;
  const input = new Float32Array(rows * K);
  const weight = new Float32Array(N * K);
  for (let i = 0; i < input.length; i++) input[i] = Math.fround(Math.sin(i + 1) * 0.5);
  for (let i = 0; i < weight.length; i++) weight[i] = Math.fround(Math.cos(i + 3) * 0.5);

  const bottomUp = witnessMatmulRowSubgroupWave64(input, rows, K, weight, N);
  const topDown = witnessMatmulRowSubgroupWave64(input, rows, K, weight, N, { reductionOrder: 'top-down' });
  assert.equal(compareF32Bits(bottomUp, bottomUp).allExact, true);
  assert.equal(compareF32Bits(topDown, bottomUp).allExact, false);
});

test('witnessMatmulF16wF32aTiled16 and witnessMatmulF16wF32aSeq distinguish tiled16, seqFma, and seqMulAdd', () => {
  const rows = 1;
  const K = 256;
  const N = 4;
  const input = new Float32Array(rows * K);
  const weight = new Float32Array(N * K);
  for (let i = 0; i < input.length; i++) input[i] = Math.fround(((i * 17) % 101 - 50) * 0.015625 + 1e-4);
  for (let i = 0; i < weight.length; i++) weight[i] = Math.fround(((i * 31) % 89 - 44) * 0.03125 - 1e-4);

  const tiled16 = witnessMatmulF16wF32aTiled16(input, rows, K, weight, N);
  const seqFma = witnessMatmulF16wF32aSeq(input, rows, K, weight, N, { useFma: true });
  const seqMulAdd = witnessMatmulF16wF32aSeq(input, rows, K, weight, N, { useFma: false });
  assert.equal(compareF32Bits(tiled16, seqFma).allExact, false);
  assert.equal(compareF32Bits(seqFma, seqMulAdd).allExact, false);
});

test('witnessGatedDeltaCore and HC activation/branch/inject witnesses reject negative controls', () => {
  const numTokens = 2;
  const convDim = 24;
  const kernelSize = 4;
  const numVHeads = 2;
  const headKDim = 4;
  const headVDim = 4;
  const qSize = 4;
  const kSize = 4;
  const valueDim = 8;
  const qRep = 2;

  const qkv = new Float32Array(numTokens * convDim);
  const zProj = new Float32Array(numTokens * valueDim);
  const aProj = new Float32Array(numTokens * numVHeads);
  const bProj = new Float32Array(numTokens * numVHeads);
  const convWeight = new Float32Array(convDim * kernelSize);
  const dtBias = new Float32Array(numVHeads);
  const aLog = new Float32Array(numVHeads);
  const normWeight = new Float32Array(headVDim);

  for (let i = 0; i < qkv.length; i++) qkv[i] = Math.fround((i + 1) * 0.05);
  for (let i = 0; i < zProj.length; i++) zProj[i] = Math.fround((i - 3) * 0.125);
  for (let i = 0; i < aProj.length; i++) aProj[i] = Math.fround(0.2 * (i + 1));
  for (let i = 0; i < bProj.length; i++) bProj[i] = Math.fround(-0.1 * (i + 1));
  for (let i = 0; i < convWeight.length; i++) convWeight[i] = Math.fround(0.15 * ((i % 5) - 2) + 0.01);
  for (let i = 0; i < numVHeads; i++) {
    dtBias[i] = Math.fround(0.05 * (i + 1));
    aLog[i] = Math.fround(-0.3 * (i + 1));
  }
  for (let i = 0; i < headVDim; i++) normWeight[i] = Math.fround(0.9 + 0.02 * i);

  const opts = { convDim, kernelSize, numVHeads, headKDim, headVDim, qSize, kSize, valueDim, qRep };
  const ref = witnessGatedDeltaCore(qkv, zProj, aProj, bProj, convWeight, dtBias, aLog, normWeight, numTokens, opts);
  const wrongGate = witnessGatedDeltaCore(qkv, zProj, aProj, bProj, convWeight, dtBias, aLog, normWeight, numTokens, {
    ...opts,
    gateActivation: 'silu',
  });
  assert.equal(compareF32Bits(wrongGate, ref).allExact, false);

  const lr = witnessLowRankSilu(new Float32Array([4.0, -4.0]), 4);
  const mix = witnessMixSigmoid(new Float32Array([0.0, 1.0]));
  const gate = witnessInjectGate(new Float32Array([0.0, 4.0, -4.0, 2.0]), 4);
  assert.equal(mix[0], 0.5);
  assert.equal(gate[0], 1.0);
  assert.ok(lr[0] > 0 && lr[1] < 0);

  const normed = new Float32Array(8).fill(1.0);
  const mix8 = new Float32Array(8).fill(0.5);
  const branch = witnessHyperConnectionBranch(normed, mix8, 1, 4, 2);
  assert.deepEqual(Array.from(branch), [0.5, 0.5]);
  const injected = witnessHyperConnectionInject(normed, branch, new Float32Array([1, 2, 1, 2]), 1, 4, 2);
  assert.deepEqual(Array.from(injected), [1.5, 1.5, 2.0, 2.0, 1.5, 1.5, 2.0, 2.0]);

  const pair = exactDyadicPairMargin([1.0, 2.0], [0.5, 0.25], [0.25, 0.25]);
  assert.equal(pair.margin, 0.25);
});

test('layer25-prerouter-witness protocol, results, and negative-control artifacts satisfy hash and 0-ULP invariants', async () => {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const crypto = await import('node:crypto');
  const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
  const repoRoot = path.resolve(new URL('..', import.meta.url).pathname);

  const protocolBytes = fs.readFileSync(
    path.join(repoRoot, 'reports/numerical-discontinuity-20261109/layer25-prerouter-witness-protocol.json')
  );
  const resultsBytes = fs.readFileSync(
    path.join(repoRoot, 'reports/numerical-discontinuity-20261109/layer25-prerouter-witness-results.json')
  );
  const negBytes = fs.readFileSync(
    path.join(repoRoot, 'reports/numerical-discontinuity-20261109/layer25-prerouter-witness-negative-result.json')
  );

  const protocol = JSON.parse(protocolBytes);
  const results = JSON.parse(resultsBytes);
  const neg = JSON.parse(negBytes);

  assert.equal(protocol.qualificationAuthority, false);
  assert.equal(results.qualified, false);
  assert.equal(results.protocolSha256, sha(protocolBytes));
  assert.equal(results.toolSha256, sha(fs.readFileSync(path.join(repoRoot, 'tools/analyze-layer25-prerouter-witness.js'))));
  assert.equal(results.witnessCoreSha256, sha(fs.readFileSync(path.join(repoRoot, 'tools/layer25-wgsl-witness-core.js'))));
  assert.equal(results.arithmeticCoreSha256, sha(fs.readFileSync(path.join(repoRoot, 'tools/numerical-discontinuity-core.js'))));

  assert.equal(results.summary.stageWitnessesChecked, 27);
  assert.equal(results.summary.stageWitnessesAll0Ulp, true);
  assert.equal(results.summary.stagePositionValuesChecked, 357456);
  assert.equal(results.summary.stagePositionValuesExact0Ulp, 357456);
  assert.equal(results.summary.fullSeqLinearCoreValuesChecked, 14155776);
  assert.equal(results.summary.fullSeqLinearCoreValuesExact0Ulp, 14155776);
  assert.equal(results.summary.fullSeqRouterLogitsValuesChecked, 1179648);
  assert.equal(results.summary.fullSeqRouterLogitsValuesExact0Ulp, 1179648);
  assert.equal(results.summary.chainedEndToEndBranchInputExact0Ulp, 7680);
  assert.equal(results.summary.chainedEndToEndRouterLogitsExact0Ulp, 1536);
  assert.equal(results.summary.faultControlsTested, 7);
  assert.equal(results.summary.faultControlsRejected, 7);
  assert.equal(results.summary.streamResidualPathAloneFlips1717, true);
  assert.equal(results.summary.incomingLayer24DriftThroughLinearAttnFlipsBothPositions, true);
  assert.equal(results.summary.incomingLayer24DriftNecessaryForGpuLayer25Flip, true);

  assert.equal(neg.exitCode, 1);
  assert.equal(neg.expectedMessage, true);
  assert.equal(neg.outputCreated, false);
});

