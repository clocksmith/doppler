// Development benchmark. Both trainers receive the same ordered token windows.
import { loadReference } from './reference.js';
import { checkTinyAutograd } from './tiny-oracle.js';
import { createTabGPTTrainer } from '../../src/training-browser.js';

function measureBuffers(device) {
  let bytes = 0;
  let peak = 0;
  const create = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = create(descriptor);
    const size = buffer.size;
    bytes += size;
    peak = Math.max(peak, bytes);
    let live = true;
    const destroy = buffer.destroy.bind(buffer);
    buffer.destroy = () => {
      if (live) bytes -= size;
      live = false;
      destroy();
    };
    return buffer;
  };
  return () => peak;
}

function stockTrainer(reference, device, cfg) {
  const stock = reference.createModel(device, cfg);
  return {
    numParams: stock.numParams,
    parameterNames: stock.paramRecords.map(record => record.name),
    trainStep: (x, y) => stock.trainStep(x, y, true),
    evaluate: (x, y) => stock.valStep(x, y),
    readParameter: name => stock.readParam(name),
  };
}

async function validationLoss(model, batches) {
  let sum = 0;
  for (const { x, y } of batches) {
    sum += await model.evaluate(x, y);
  }
  return sum / batches.length;
}

async function trajectory(device, peakBytes, ref, cfg, spec, vendor) {
  const start = performance.now();
  const model = vendor === 'doppler' ? await createTabGPTTrainer(device, cfg) : stockTrainer(ref, device, cfg);
  await device.queue.onSubmittedWorkDone();
  const buildMs = performance.now() - start;
  const validationBatches = spec.validationSeeds.map(seed => ref.batch('val', seed, cfg.B, cfg.T));
  const validationBefore = await validationLoss(model, validationBatches);
  const steps = [];
  for (let i = 0; i < spec.warmupSteps + spec.timedSteps; i++) {
    const { x, y } = ref.batch('train', spec.firstBatchSeed + i, cfg.B, cfg.T);
    const begin = performance.now();
    const loss = await model.trainStep(x, y);
    await device.queue.onSubmittedWorkDone();
    steps.push({ loss, ms: performance.now() - begin });
  }
  const validationAfter = await validationLoss(model, validationBatches);
  if (![validationBefore, validationAfter, ...steps.map(step => step.loss)].every(Number.isFinite)) {
    throw new Error(vendor + ' produced non-finite loss');
  }
  const timedMs = steps.slice(spec.warmupSteps).reduce((sum, step) => sum + step.ms, 0);
  return {
    numParams: model.numParams, precision: cfg.f16 ? 'mixed-f16' : 'f32', buildMs, validationBefore, validationAfter, steps,
    tokensPerSecond: spec.timedSteps * cfg.B * cfg.T * 1000 / timedMs,
    peakRequestedGpuBufferBytes: peakBytes(),
  };
}

function updateError(beforeStock, beforeNative, expected, actual) {
  if (beforeStock.length !== actual.length || beforeNative.length !== actual.length || actual.length !== expected.length) throw new Error('parameter shape mismatch');
  let expectedSum = 0;
  let errorSum = 0;
  let maxAbsError = 0;
  for (let i = 0; i < beforeStock.length; i++) {
    const delta = expected[i] - beforeStock[i];
    const error = (actual[i] - beforeNative[i]) - delta;
    expectedSum += delta * delta;
    errorSum += error * error;
    maxAbsError = Math.max(maxAbsError, Math.abs(error));
  }
  return { referenceRms: Math.sqrt(expectedSum / beforeStock.length), errorRms: Math.sqrt(errorSum / beforeStock.length), maxAbsError };
}

async function parity(device, ref, cfg, spec) {
  const stock = stockTrainer(ref, device, cfg);
  const native = await createTabGPTTrainer(device, cfg);
  if (stock.numParams !== native.numParams || stock.parameterNames.length !== native.parameterNames.length) {
    throw new Error('parameter count mismatch');
  }
  const initial = new Map();
  for (const name of stock.parameterNames) {
    const expected = await stock.readParameter(name);
    const actual = await native.readParameter(name);
    if (expected.length !== actual.length || expected.some((v, i) => !Number.isFinite(v) || v !== actual[i])) {
      throw new Error('initial weight mismatch: ' + name);
    }
    initial.set(name, { stock: expected, native: actual });
  }
  const steps = [];
  for (let step = 0; step < 2; step++) {
    const { x, y } = ref.batch('train', spec.firstBatchSeed + step, cfg.B, cfg.T);
    const stockLoss = await stock.trainStep(x, y);
    const nativeLoss = await native.trainStep(x, y);
    const parameters = {};
    for (const [name, before] of initial) {
      const expected = await stock.readParameter(name);
      const actual = await native.readParameter(name);
      const error = updateError(before.stock, before.native, expected, actual);
      parameters[name] = { ...error, pass: error.errorRms <= spec.updateRmsAbsTolerance + spec.updateRmsRelativeTolerance * error.referenceRms };
      if (step === 0) initial.set(name, { stock: expected, native: actual });
    }
    const pass = Math.abs(stockLoss - nativeLoss) <= spec.lossAbsTolerance && Object.values(parameters).every(x => x.pass);
    steps.push({ stockLoss, nativeLoss, pass, parameters });
  }
  return { numParams: stock.numParams, tensors: initial.size, allInitialWeightsIdentical: true, steps, pass: steps.every(step => step.pass) };
}

export async function runBrowser(mode, spec) {
  const ref = await loadReference('/reference-assets/');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info.isFallbackAdapter) {
    throw new Error('Physical WebGPU adapter required');
  }
  const gpu = Object.fromEntries(['vendor','architecture','device','description'].map(k => [k, adapter.info[k]]));
  if ((!gpu.vendor && !gpu.architecture) || /swiftshader|llvmpipe|software/i.test(JSON.stringify(gpu))) throw new Error('Unidentified physical GPU');
  const requiredLimits = Object.fromEntries(['maxBufferSize', 'maxStorageBufferBindingSize'].map(k => [k, adapter.limits[k]]));
  const requiredFeatures = mode === 'stock-mixed' ? ['shader-f16'] : [];
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  const errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const peak = measureBuffers(device);
  try {
    let result;
    if (mode === 'oracle') {
      const create = cfg => createTabGPTTrainer(device, cfg);
      const aligned = await checkTinyAutograd(create);
      const tails = await checkTinyAutograd(create, new URL('./tiny-pytorch-odd-oracle.json', import.meta.url));
      result = { aligned, tails, pass: aligned.pass && tails.pass };
    } else if (mode === 'parity') {
      result = await parity(device, ref, spec.model, spec);
    } else {
      const cfg = mode === 'stock-mixed' ? { ...spec.model, f16: true } : spec.model;
      result = await trajectory(device, peak, ref, cfg, spec, mode);
    }
    await device.queue.onSubmittedWorkDone();
    if (errors.length) throw new Error(errors.slice(0, 3).join('\n'));
    return { gpu, provenance: ref.provenance, result };
  } finally {
    device.destroy();
  }
}
