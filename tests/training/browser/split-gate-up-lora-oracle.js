import { loadBackwardRegistry } from '../../../src/config/backward-registry-loader.js';
import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { setTrainingConfig } from '../../../src/config/training-defaults.js';
import { AutogradTape, OpType } from '../../../src/experimental/training/autograd.js';
import { LoraAdapter } from '../../../src/experimental/training/lora.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { runMatmul, runResidualAdd, runSiLU } from '../../../src/gpu/kernels/index.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function makeTensor(values, dtype, shape, label) {
  const byteLength = Math.ceil(values.byteLength / 4) * 4;
  const upload = byteLength === values.byteLength
    ? values
    : (() => {
        const padded = new Uint8Array(byteLength);
        padded.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
        return padded;
      })();
  const buffer = acquireBuffer(byteLength, undefined, label);
  uploadData(buffer, upload);
  return createTensor(buffer, dtype, shape, label);
}

function matmul(left, right, rows, inner, cols) {
  const output = new Float32Array(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      let sum = 0;
      for (let index = 0; index < inner; index += 1) {
        sum += left[(row * inner) + index] * right[(index * cols) + col];
      }
      output[(row * cols) + col] = sum;
    }
  }
  return output;
}

function transpose(values, rows, cols) {
  const output = new Float32Array(values.length);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      output[(col * rows) + row] = values[(row * cols) + col];
    }
  }
  return output;
}

function add(left, right) {
  return Float32Array.from(left, (value, index) => value + right[index]);
}

function scale(values, factor) {
  return Float32Array.from(values, (value) => value * factor);
}

function compare(actual, expected) {
  let maxAbsError = 0;
  let squaredError = 0;
  let allFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    allFinite = allFinite && Number.isFinite(actual[index]);
  }
  return {
    elementCount: expected.length,
    allFinite,
    maxAbsError,
    rmse: Math.sqrt(squaredError / expected.length),
  };
}

function cpuOracle(input, gateWeight, upWeight, gateA, gateB, upA, upB, gradOutput, dims) {
  const { M, K, N, rank, adapterScale } = dims;
  const gateDown = matmul(input, gateA, M, K, rank);
  const upDown = matmul(input, upA, M, K, rank);
  const baseGate = matmul(input, gateWeight, M, K, N);
  const baseUp = matmul(input, upWeight, M, K, N);
  const gateDelta = scale(matmul(gateDown, gateB, M, rank, N), adapterScale);
  const upDelta = scale(matmul(upDown, upB, M, rank, N), adapterScale);
  const gate = add(baseGate, gateDelta);
  const up = add(baseUp, upDelta);
  const activated = new Float32Array(M * N);
  const gradGate = new Float32Array(M * N);
  const gradUp = new Float32Array(M * N);
  for (let index = 0; index < activated.length; index += 1) {
    const clamped = Math.max(-15, Math.min(15, gate[index]));
    const sigmoid = 1 / (1 + Math.exp(-clamped));
    const silu = gate[index] * sigmoid;
    activated[index] = silu * up[index];
    gradGate[index] = gradOutput[index]
      * sigmoid
      * (1 + (gate[index] * (1 - sigmoid)))
      * up[index];
    gradUp[index] = gradOutput[index] * silu;
  }
  const gateBGrad = scale(
    matmul(transpose(gateDown, M, rank), gradGate, rank, M, N),
    adapterScale
  );
  const upBGrad = scale(
    matmul(transpose(upDown, M, rank), gradUp, rank, M, N),
    adapterScale
  );
  const gateDownGrad = scale(
    matmul(gradGate, transpose(gateB, rank, N), M, N, rank),
    adapterScale
  );
  const upDownGrad = scale(
    matmul(gradUp, transpose(upB, rank, N), M, N, rank),
    adapterScale
  );
  const gateAGrad = matmul(transpose(input, M, K), gateDownGrad, K, M, rank);
  const upAGrad = matmul(transpose(input, M, K), upDownGrad, K, M, rank);
  return {
    baseGate,
    baseUp,
    gateDown,
    upDown,
    gateDelta,
    upDelta,
    gate,
    up,
    activated,
    gateAGrad,
    gateBGrad,
    upAGrad,
    upBGrad,
  };
}

async function readF32(tensor) {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  return new Float32Array(await readBuffer(tensor.buffer, count * Float32Array.BYTES_PER_ELEMENT));
}

async function executeCase(gateBValues, label) {
  const dims = { M: 2, K: 4, N: 3, rank: 2, alpha: 4 };
  dims.adapterScale = dims.alpha / dims.rank;
  const inputValues = new Float32Array([
    0.2, -0.4, 0.7, 0.1,
    -0.3, 0.6, 0.5, -0.2,
  ]);
  const gateWeightBits = f32ToF16Array(new Float32Array([
    0.25, -0.1, 0.3,
    -0.2, 0.4, 0.15,
    0.05, -0.35, 0.2,
    0.3, 0.1, -0.25,
  ]));
  const upWeightBits = f32ToF16Array(new Float32Array([
    -0.15, 0.2, 0.35,
    0.45, -0.3, 0.05,
    -0.25, 0.1, 0.4,
    0.2, 0.3, -0.1,
  ]));
  const gateAValues = new Float32Array([0.08, -0.04, 0.03, 0.07, -0.06, 0.05, 0.09, 0.02]);
  const upAValues = new Float32Array([-0.05, 0.06, 0.04, -0.03, 0.07, 0.08, -0.02, 0.09]);
  const upBValues = new Float32Array([0.11, -0.07, 0.05, -0.04, 0.08, 0.12]);
  const gradOutputValues = new Float32Array([0.4, -0.6, 0.25, -0.35, 0.5, 0.7]);
  const input = makeTensor(inputValues, 'f32', [dims.M, dims.K], `${label}_input`);
  const gateWeight = makeTensor(gateWeightBits, 'f16', [dims.K, dims.N], `${label}_gate_weight`);
  const upWeight = makeTensor(upWeightBits, 'f16', [dims.K, dims.N], `${label}_up_weight`);
  const gradOutput = makeTensor(gradOutputValues, 'f32', [dims.M, dims.N], `${label}_grad_output`);
  const gateAdapter = new LoraAdapter({ inDim: dims.K, outDim: dims.N, rank: dims.rank, alpha: dims.alpha, dtype: 'f32' });
  const upAdapter = new LoraAdapter({ inDim: dims.K, outDim: dims.N, rank: dims.rank, alpha: dims.alpha, dtype: 'f32' });
  uploadData(gateAdapter.A.buffer, gateAValues);
  uploadData(gateAdapter.B.buffer, gateBValues);
  uploadData(upAdapter.A.buffer, upAValues);
  uploadData(upAdapter.B.buffer, upBValues);
  const tape = new AutogradTape(loadBackwardRegistry());
  try {
    const baseOptions = {
      M: dims.M,
      N: dims.N,
      K: dims.K,
      transposeB: false,
      stopGradInputs: [1],
    };
    const baseGate = await tape.record(
      OpType.MATMUL,
      (x, weight) => runMatmul(x, weight, dims.M, dims.N, dims.K, {
        transposeB: false,
        outputDtype: 'f32',
      }),
      [input, gateWeight],
      baseOptions
    );
    const baseUp = await tape.record(
      OpType.MATMUL,
      (x, weight) => runMatmul(x, weight, dims.M, dims.N, dims.K, {
        transposeB: false,
        outputDtype: 'f32',
      }),
      [input, upWeight],
      baseOptions
    );
    const gateRecordStart = tape.records.length;
    const gateDelta = await gateAdapter.forward(input, tape);
    const gateDown = tape.records[gateRecordStart].output;
    const gateUpRaw = tape.records[gateRecordStart + 1].output;
    const upRecordStart = tape.records.length;
    const upDelta = await upAdapter.forward(input, tape);
    const upDown = tape.records[upRecordStart].output;
    const upUpRaw = tape.records[upRecordStart + 1].output;
    const gate = await tape.record(
      OpType.RESIDUAL_ADD,
      (base, delta) => runResidualAdd(base, delta, dims.M * dims.N),
      [baseGate, gateDelta],
      { size: dims.M * dims.N }
    );
    const up = await tape.record(
      OpType.RESIDUAL_ADD,
      (base, delta) => runResidualAdd(base, delta, dims.M * dims.N),
      [baseUp, upDelta],
      { size: dims.M * dims.N }
    );
    const activated = await tape.record(
      OpType.SILU_GATED,
      (gateInput, upInput) => runSiLU(upInput, {
        size: dims.M * dims.N,
        gate: gateInput,
        inputActivation: 'identity',
        swigluLimit: null,
      }),
      [gate, up],
      { count: dims.M * dims.N, swigluLimit: 0 }
    );
    const grads = await tape.backward(new Map([[activated, gradOutput]]));
    const actual = {
      gateA: await readF32(gateAdapter.A),
      gateB: await readF32(gateAdapter.B),
      upA: await readF32(upAdapter.A),
      upB: await readF32(upAdapter.B),
      baseGate: await readF32(baseGate),
      baseUp: await readF32(baseUp),
      gateDown: await readF32(gateDown),
      upDown: await readF32(upDown),
      gateUpRaw: await readF32(gateUpRaw),
      upUpRaw: await readF32(upUpRaw),
      gateDelta: await readF32(gateDelta),
      upDelta: await readF32(upDelta),
      gate: await readF32(gate),
      up: await readF32(up),
      activated: await readF32(activated),
      gateAGrad: await readF32(grads.get(gateAdapter.A)),
      gateBGrad: await readF32(grads.get(gateAdapter.B)),
      upAGrad: await readF32(grads.get(upAdapter.A)),
      upBGrad: await readF32(grads.get(upAdapter.B)),
    };
    const expected = cpuOracle(
      inputValues,
      f16ToF32Array(gateWeightBits),
      f16ToF32Array(upWeightBits),
      gateAValues,
      gateBValues,
      upAValues,
      upBValues,
      gradOutputValues,
      dims
    );
    expected.gateA = gateAValues;
    expected.gateB = gateBValues;
    expected.upA = upAValues;
    expected.upB = upBValues;
    expected.gateUpRaw = scale(expected.gateDelta, 1 / dims.adapterScale);
    expected.upUpRaw = scale(expected.upDelta, 1 / dims.adapterScale);
    return {
      actual,
      expected,
      comparisons: Object.fromEntries(
        Object.keys(expected).map((key) => [key, compare(actual[key], expected[key])])
      ),
    };
  } finally {
    gateAdapter.dispose();
    upAdapter.dispose();
    releaseBuffer(input.buffer);
    releaseBuffer(gateWeight.buffer);
    releaseBuffer(upWeight.buffer);
    releaseBuffer(gradOutput.buffer);
  }
}

export async function runSplitGateUpLoraOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  setTrainingConfig({ training: { precision: { loraParams: 'f32' } } });
  await initDevice();

  const tolerance = 5e-5;
  const baselineGateB = new Float32Array([0.1, -0.05, 0.09, 0.06, -0.08, 0.04]);
  const baseline = await executeCase(baselineGateB, 'split_gate_up_baseline');
  const perturbedGateB = new Float32Array(baselineGateB);
  perturbedGateB[2] += 0.125;
  const perturbed = await executeCase(perturbedGateB, 'split_gate_up_perturbed');
  const perturbation = compare(perturbed.actual.activated, baseline.actual.activated);
  const comparisons = baseline.comparisons;
  const gradientKeys = ['gateAGrad', 'gateBGrad', 'upAGrad', 'upBGrad'];
  const passed = Object.values(comparisons).every(
    (entry) => entry.allFinite && entry.maxAbsError <= tolerance
  ) && gradientKeys.every(
    (key) => baseline.actual[key].some((value) => value !== 0)
  ) && perturbation.maxAbsError > 1e-4;
  const capabilities = getKernelCapabilities();
  return {
    artifactType: 'split_gate_up_lora_backward_oracle',
    schemaVersion: 1,
    passed,
    dimensions: { M: 2, K: 4, N: 3, rank: 2, alpha: 4 },
    tolerance: { maxAbsError: tolerance },
    comparisons,
    nonzeroGradientFamilies: Object.fromEntries(
      gradientKeys.map((key) => [key, baseline.actual[key].some((value) => value !== 0)])
    ),
    negativeControl: {
      perturbation: 'gate_proj_lora_b_index_2_plus_0.125',
      activatedOutputDifference: perturbation,
      passed: perturbation.maxAbsError > 1e-4,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Split gate_proj/up_proj LoRA and gated-SiLU block mechanics only; not a Qwen layer, optimizer update, capability, or backend parity receipt.',
  };
}
