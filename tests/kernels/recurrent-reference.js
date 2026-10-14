import assert from 'node:assert/strict';

// Independent scalar Float64 equations, not a transcription of WGSL reductions.
// Test-only: these arrays must never be used by production inference.
export const RECURRENT_REFERENCE = Object.freeze({
  model: 'Qwen/Qwen3.5-0.8B',
  modelRevision: '2fc06364715b967f1860aea9cf38778875588b17',
  equations: 'https://github.com/huggingface/transformers/blob/0a896aa41bba78f92338db21cc468fe043888657/src/transformers/models/qwen3_5/modeling_qwen3_5.py',
  sourceSha256: 'f3fecce29dee6cea1ac951c947a538fdcf01abfc21f8f0e33a78b535fe6837c3',
  functions: ['l2norm', 'torch_recurrent_gated_delta_rule', 'Qwen3_5RMSNormGated'],
});

// The sigmoid gate follows the pinned qwen4_exp source retained with the
// Flash-Next reference receipts; only RMSNormGated's activation differs there.
export const SIGMOID_NORM_GATE_REFERENCE = Object.freeze({
  model: 'Qwen/Qwen3.8-Flash-Next',
  modelRevision: 'de4b8e4d43b917e7706784d8bb445c9af86a3540',
  equations: 'https://raw.githubusercontent.com/huggingface/transformers/e0eb2810ac1b2caf6d082f247d8d37162ecf67a0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py',
  sourceSha256: '87c27c88424cea73d5d8d9179f76a77a97e6e4249ce43ecff79a8b255fab0504',
  functions: ['Qwen4ExpTextRMSNormGated'],
  receipt: 'artifacts/qwen38-flash-next-20261013/linear-attention-reference.json',
});

const sigmoid64 = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));

// Gated RMSNorm gate act(z): Qwen3_5RMSNormGated uses silu = z * sigmoid(z);
// Qwen4ExpTextRMSNormGated with output_gate_type="sigmoid" uses sigmoid(z).
export function normGate(z, normGateActivation) {
  if (normGateActivation === 'silu') return z * sigmoid64(z);
  if (normGateActivation === 'sigmoid') return sigmoid64(z);
  throw new Error(`Unsupported norm-gate activation ${JSON.stringify(normGateActivation)}`);
}

export function recurrentTraceLayout(p) {
  const heads = p.numTokens * p.numVHeads;
  const vectors = heads * p.headVDim;
  const matrices = vectors * p.headKDim;
  const sizes = {
    qScale: heads, kScale: heads, beta: heads, logDecay: heads, decay: heads,
    normalizedQ: heads * p.headKDim, normalizedK: heads * p.headKDim,
    decayedState: matrices, memory: vectors, correction: vectors,
    updatedState: matrices, rawOutput: vectors, invRms: heads,
    gate: vectors, gatedOutput: vectors,
  };
  let length = 0;
  const fields = Object.fromEntries(Object.entries(sizes).map(([name, size]) => {
    const field = { offset: length, length: size }; length += size; return [name, field];
  }));
  return { fields, length };
}

export function decodeCapturedTensor(record, role) {
  const matches = record.tensors.filter(t => t.role === role);
  assert.equal(matches.length, 1, `Expected exactly one ${role} tensor`);
  const bytes = Buffer.from(matches[0].data, 'base64');
  assert.equal(bytes.length, matches[0].bytes);
  assert.equal(bytes.length % 4, 0);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
}

export function pairLinearCaptures(capture) {
  return capture.captures.flatMap((group, captureIndex) => {
    assert.deepEqual(group.errors, [], 'Capture contains failed observations');
    const pairs = new Map();
    for (const record of group.records.filter(r => /^linear-(inputs|outputs)$/.test(r.boundary))) {
      const { layerIdx, step, prompt, dispatch } = record;
      assert(Number.isInteger(layerIdx) && Number.isInteger(step) &&
        Number.isInteger(prompt) && Number.isInteger(dispatch), 'Missing linear capture coordinates');
      const key = JSON.stringify([captureIndex, layerIdx, prompt, step, dispatch]);
      const pair = pairs.get(key) ?? { key, coordinate: { captureIndex, layerIdx, prompt, step, dispatch } };
      const side = record.boundary === 'linear-inputs' ? 'input' : 'expected';
      assert(!pair[side], `Duplicate ${side} at ${key}`);
      pair[side] = record; pairs.set(key, pair);
    }
    for (const pair of pairs.values()) {
      assert(pair.input && pair.expected, `Unpaired capture at ${pair.key}`);
      assert.deepEqual(pair.input.params, pair.expected.params, 'Captured operation geometry changed');
    }
    return [...pairs.values()];
  });
}

export function recurrentReference(input, expected) {
  const p = input.params;
  for (const name of ['numTokens', 'numVHeads', 'numKHeads', 'headKDim', 'headVDim', 'qRep']) {
    assert(Number.isSafeInteger(p[name]) && p[name] > 0, `Invalid ${name}`);
  }
  assert.equal(p.numVHeads, p.numKHeads * p.qRep);
  assert.equal(p.valueDim, p.numVHeads * p.headVDim);
  assert.equal(p.qSize, p.numKHeads * p.headKDim);
  assert.equal(p.kSize, p.qSize);
  assert.equal(p.convDim, p.qSize + p.kSize + p.valueDim);
  assert(['shared', 'per_head'].includes(p.normMode));
  // Captures recorded before the field existed are Qwen3.5 silu lanes, the
  // same legacy meaning the manifest schema default carries.
  const normGateActivation = p.normGateActivation === undefined ? 'silu' : p.normGateActivation;
  assert(['silu', 'sigmoid'].includes(normGateActivation), 'Invalid normGateActivation');
  assert(Number.isFinite(p.qkL2NormEps) && p.qkL2NormEps >= 0);
  assert(Number.isFinite(p.rmsNormEps) && p.rmsNormEps >= 0);
  const layout = recurrentTraceLayout(p);
  assert(Number.isSafeInteger(layout.length) && layout.length * 8 <= 128 * 1024 * 1024,
    'Recurrent reference trace exceeds its 128 MiB diagnostic limit');
  const tensors = Object.fromEntries(['z', 'a', 'b', 'dtBias', 'aLog', 'normWeight', 'recurrentState']
    .map(role => [role, decodeCapturedTensor(input, role)]));
  const conv = decodeCapturedTensor(expected, 'convOutput');
  const stateSize = p.numVHeads * p.headKDim * p.headVDim;
  const state = Float64Array.from(tensors.recurrentState.subarray(0, stateSize));
  assert.equal(state.length, stateSize);
  const trace = new Float64Array(layout.length);
  const stages = Object.fromEntries(Object.entries(layout.fields)
    .map(([name, f]) => [name, trace.subarray(f.offset, f.offset + f.length)]));
  const eps = Math.fround(p.qkL2NormEps), rmsEps = Math.fround(p.rmsNormEps);
  const softplus = x => x > 20 ? x : Math.log1p(Math.exp(x));
  const sigmoid = sigmoid64;
  for (let t = 0; t < p.numTokens; t++) for (let h = 0; h < p.numVHeads; h++) {
    const row = t * p.numVHeads + h, srcHead = Math.floor(h / p.qRep);
    const qBase = t * p.convDim + srcHead * p.headKDim, kBase = qBase + p.qSize;
    const vBase = t * p.convDim + p.qSize + p.kSize + h * p.headVDim;
    let qSq = 0, kSq = 0;
    for (let k = 0; k < p.headKDim; k++) { qSq += conv[qBase + k] ** 2; kSq += conv[kBase + k] ** 2; }
    const qScale = 1 / Math.sqrt(qSq + eps) / Math.sqrt(p.headKDim);
    const kScale = 1 / Math.sqrt(kSq + eps);
    stages.qScale[row] = qScale; stages.kScale[row] = kScale;
    const beta = sigmoid(tensors.b[(p.abPacked ? p.bProjOffsetElements : 0) + row]);
    const logDecay = -Math.exp(tensors.aLog[h]) * softplus(tensors.a[row] + tensors.dtBias[h]);
    const decay = Math.exp(logDecay);
    stages.beta[row] = beta; stages.logDecay[row] = logDecay; stages.decay[row] = decay;
    const matrixBase = h * p.headKDim * p.headVDim;
    const traceBase = row * p.headKDim * p.headVDim;
    for (let k = 0; k < p.headKDim; k++) {
      stages.normalizedQ[row * p.headKDim + k] = conv[qBase + k] * qScale;
      stages.normalizedK[row * p.headKDim + k] = conv[kBase + k] * kScale;
      for (let v = 0; v < p.headVDim; v++) {
        const kv = k * p.headVDim + v;
        state[matrixBase + kv] *= decay;
        stages.decayedState[traceBase + kv] = state[matrixBase + kv];
      }
    }
    for (let v = 0; v < p.headVDim; v++) {
      const out = row * p.headVDim + v;
      let memory = 0;
      for (let k = 0; k < p.headKDim; k++) memory += state[matrixBase + k * p.headVDim + v] * stages.normalizedK[row * p.headKDim + k];
      const correction = (conv[vBase + v] - memory) * beta;
      stages.memory[out] = memory; stages.correction[out] = correction;
      let raw = 0;
      for (let k = 0; k < p.headKDim; k++) {
        const kv = k * p.headVDim + v;
        state[matrixBase + kv] += stages.normalizedK[row * p.headKDim + k] * correction;
        stages.updatedState[traceBase + kv] = state[matrixBase + kv];
        raw += state[matrixBase + kv] * stages.normalizedQ[row * p.headKDim + k];
      }
      stages.rawOutput[out] = raw;
    }
    let squareSum = 0;
    for (let v = 0; v < p.headVDim; v++) squareSum += stages.rawOutput[row * p.headVDim + v] ** 2;
    const invRms = 1 / Math.sqrt(squareSum / p.headVDim + rmsEps); stages.invRms[row] = invRms;
    for (let v = 0; v < p.headVDim; v++) {
      const out = row * p.headVDim + v;
      const z = tensors.z[p.qkvzPacked ? t * (p.convDim + p.valueDim) + p.convDim + h * p.headVDim + v : out];
      const gate = normGate(z, normGateActivation); stages.gate[out] = gate;
      stages.gatedOutput[out] = stages.rawOutput[out] * invRms * tensors.normWeight[p.normMode === 'per_head' ? h * p.headVDim + v : v] * gate;
    }
  }
  assert(trace.every(Number.isFinite), 'Non-finite independent recurrent reference');
  return { trace, layout, finalState: state, output: stages.gatedOutput };
}
