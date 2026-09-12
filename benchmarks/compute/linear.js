import { requireCondition } from './contract.js';

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function createLinearInputs(shape, shared, caseIndex) {
  const random = seededRandom(shared.seed + caseIndex);
  const values = count => Float32Array.from({ length: count }, () => (random() * 2 - 1) * shape.inputScale);
  const x = values(shape.m * shape.k);
  const weights = values(shape.k * shape.n);
  const bias = Float32Array.from({ length: shape.n }, (_, i) => ((i % 7) - 3) * shape.biasScale / 3);
  return { x, weights, bias };
}

// Independent scalar reference: no shader text, emitted plan, or GPU output is
// used to construct the answer. Accumulate in JS double precision, then model
// the unfused f32 storage boundaries before the mathematical activation.
export function referenceLinear(shape, inputs, shared) {
  const result = new Float32Array(shape.m * shape.n);
  for (let row = 0; row < shape.m; row += 1) {
    for (let column = 0; column < shape.n; column += 1) {
      let dot = 0;
      for (let inner = 0; inner < shape.k; inner += 1) {
        dot += inputs.x[row * shape.k + inner] * inputs.weights[inner * shape.n + column];
      }
      const value = Math.fround(Math.fround(dot) + inputs.bias[column]);
      const clamped = Math.max(-shared.sigmoidClamp, Math.min(shared.sigmoidClamp, value));
      result[row * shape.n + column] = value / (1 + Math.exp(-clamped));
    }
  }
  return result;
}

export function compareOutput(actual, expected, shared) {
  let failures = actual.length === expected.length + shared.guardElements ? 0 : 1;
  let maxAbs = 0;
  let maxScaledError = 0;
  let nonFinite = 0;
  let guardErrors = 0;
  const firstMismatches = [];
  for (let i = 0; i < expected.length; i += 1) {
    const finite = Number.isFinite(actual[i]);
    if (!finite) nonFinite += 1;
    const abs = finite ? Math.abs(actual[i] - expected[i]) : null;
    const budget = shared.absoluteTolerance + shared.relativeTolerance * Math.abs(expected[i]);
    if (finite) {
      maxAbs = Math.max(maxAbs, abs);
      maxScaledError = Math.max(maxScaledError, abs / budget);
    }
    if (!finite || abs > budget) {
      failures += 1;
      if (firstMismatches.length < 8) firstMismatches.push({ index: i, actual: finite ? actual[i] : String(actual[i]), expected: expected[i], budget });
    }
  }
  for (let i = expected.length; i < actual.length; i += 1) {
    if (actual[i] !== shared.guardValue) guardErrors += 1;
  }
  return { passed: failures === 0 && guardErrors === 0, failures, nonFinite, guardErrors, maxAbs, maxScaledError, firstMismatches };
}

export function runOracleControls(expected, shared) {
  const valid = new Float32Array(expected.length + shared.guardElements);
  valid.set(expected);
  valid.fill(shared.guardValue, expected.length);
  const controls = [];
  function record(id, actual, shouldPass) {
    const observed = compareOutput(actual, expected, shared);
    requireCondition(observed.passed === shouldPass, `Numerical oracle control failed: ${id}`);
    controls.push({ id, expectedPass: shouldPass, observed });
  }
  record('known-correct', valid, true);
  const wrong = valid.slice();
  wrong[0] += 1 + Math.abs(expected[0]);
  record('wrong-element', wrong, false);
  const missing = valid.slice();
  missing[Math.floor(expected.length / 2)] = NaN;
  record('unwritten-element', missing, false);
  const nonFinite = valid.slice();
  nonFinite[0] = Infinity;
  record('non-finite-element', nonFinite, false);
  record('truncated-output', valid.slice(0, expected.length - 1), false);
  const guard = valid.slice();
  guard[expected.length] = 0;
  record('out-of-range-write', guard, false);
  return controls;
}

export function createLinearPlan(shape, lane, suite) {
  const { m, n, k } = shape;
  const workgroupSize = suite.engine.workgroupSize;
  const elements = m * n;
  const outputBytes = (elements + suite.shared.guardElements) * 4;
  const resources = {
    x: { bytes: m * k * 4, kind: 'storage' },
    weights: { bytes: k * n * 4, kind: 'storage' },
    bias: { bytes: n * 4, kind: 'storage' },
    dims: { bytes: 16, kind: 'uniform' },
    output: { bytes: outputBytes, kind: 'storage' },
  };
  if (!lane.fused) {
    resources.intermediate = { bytes: outputBytes, kind: 'storage' };
    resources.biasDims = { bytes: 32, kind: 'uniform' };
    resources.siluDims = { bytes: 16, kind: 'uniform' };
  }
  const bind = (resource, type) => ({ resource, type });
  const matmul = {
    id: 'matmul', source: 'benchmarks/compute/linear.wgsl', entryPoint: 'main',
    constants: { WORKGROUP_SIZE: workgroupSize, FUSE_EPILOGUE: Number(lane.fused), SIGMOID_CLAMP: suite.shared.sigmoidClamp },
    bindings: [bind('dims', 'uniform'), bind('x', 'read-only-storage'), bind('weights', 'read-only-storage'), bind('bias', 'read-only-storage'), bind(lane.fused ? 'output' : 'intermediate', 'storage')],
    dispatch: [Math.ceil(elements / workgroupSize), 1, 1],
  };
  const steps = [matmul];
  if (!lane.fused) {
    steps.push({ id: 'bias', source: 'src/gpu/kernels/bias_add.wgsl', entryPoint: 'main', constants: { WORKGROUP_SIZE: workgroupSize },
      bindings: [bind('biasDims', 'uniform'), bind('intermediate', 'storage'), bind('bias', 'read-only-storage')],
      dispatch: [Math.ceil(n / workgroupSize), m, 1] });
    steps.push({ id: 'silu', source: 'src/gpu/kernels/silu.wgsl', entryPoint: 'main',
      constants: { WORKGROUP_SIZE: workgroupSize, HAS_GATE: 0, GATE_USE_SIGMOID: 0, INPUT_USE_IDENTITY: 0, USE_SPLIT: 0, USE_VEC4: 0, USE_ROWSPLIT: 0 },
      bindings: [bind('siluDims', 'uniform'), bind('intermediate', 'read-only-storage'), bind('output', 'storage'), bind('bias', 'read-only-storage')],
      dispatch: [Math.ceil(elements / workgroupSize), 1, 1] });
  }
  return {
    id: lane.id, semanticContract: { operation: 'silu(matmul(x,weights)+bias)', m, n, k, dtype: 'f32', sigmoidClamp: suite.shared.sigmoidClamp },
    submission: lane.submission, resources, steps, output: 'output', outputElements: elements, outputBytes,
    resetResources: lane.fused ? ['output'] : ['output', 'intermediate'],
    logicalIntermediateBytesPerOperation: lane.fused ? 0 : elements * 4 * 4,
    trafficNote: 'Analytical intermediate stores/loads only; not measured DRAM traffic or bandwidth.',
  };
}

export function linearResourceData(shape, inputs) {
  return { ...inputs, dims: new Uint32Array([shape.m, shape.n, shape.k, 0]),
    biasDims: new Uint32Array([shape.m, shape.n, 0, 0, shape.m, 0, 0, 0]),
    siluDims: new Uint32Array([shape.m * shape.n, shape.m * shape.n, 0, 0]) };
}
