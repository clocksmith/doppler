/** Replay captured linear-attention operands. Diagnostic only; no model acceptance. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { buildLinearActivationDiagnostic } from '../kernels/linear-activation-diagnostic.js';
import { decodeCapturedTensor, pairLinearCaptures, recurrentReference, RECURRENT_REFERENCE } from '../kernels/recurrent-reference.js';
import { observeRecurrentShader } from '../kernels/recurrent-observation.js';
import { interveneRecurrentShader, recurrentInterventionOperands } from '../kernels/recurrent-intervention.js';
import { buildRecurrentAccumulationDiagnostic } from '../kernels/recurrent-accumulation-diagnostic.js';

const [capturePath, installedPackage, reploidRoot, output, operation, captureKey] = process.argv.slice(2);
assert(capturePath && installedPackage && reploidRoot && output && process.env.REPLOID_EXECUTOR_WS);
assert(['conv', 'recurrent'].includes(operation));
const bytes = await readFile(capturePath), capture = JSON.parse(bytes);
const pairs = pairLinearCaptures(capture).filter(p => !captureKey || p.key === captureKey);
assert.equal(pairs.length, 1, 'Supply the exact capture key when more than one layer/step is present');
const { input, expected, coordinate } = pairs[0];
const independent = operation === 'recurrent' ? recurrentReference(input, expected) : null;
const originalShader = await readFile(resolve(installedPackage, `src/gpu/kernels/gated_delta_${operation}.wgsl`), 'utf8');
const observeActivation = process.env.DOPPLER_LINEAR_OBSERVE_ACTIVATION === '1';
const activationCandidate = process.env.DOPPLER_LINEAR_ACTIVATION_DIAGNOSTIC === '1';
const macStrictMath = process.env.DOPPLER_MAC_STRICT_MATH_DIAGNOSTIC === '1';
const observeRecurrent = process.env.DOPPLER_LINEAR_OBSERVE_RECURRENT === '1';
const observedStages = process.env.DOPPLER_LINEAR_RECURRENT_STAGES?.split(',') ?? Object.keys(independent?.layout.fields ?? {});
const replayPrefixes = process.env.DOPPLER_LINEAR_RECURRENT_PREFIXES === '1';
const intervention = process.env.DOPPLER_LINEAR_RECURRENT_INTERVENTION || null;
const compensated = process.env.DOPPLER_LINEAR_COMPENSATED_DIAGNOSTIC === '1';
if (compensated) {
  assert.equal(process.env.DOPPLER_TEST_ONLY_ARITHMETIC, '1', 'Compensated experiment requires DOPPLER_TEST_ONLY_ARITHMETIC=1');
  assert(operation === 'recurrent' && !observeRecurrent && !replayPrefixes && !intervention);
}
const baselinePath = process.env.DOPPLER_LINEAR_BASELINE;
assert(!observeRecurrent || baselinePath, 'Intermediate observation requires an uninstrumented baseline receipt');
const baselineBytes = baselinePath ? await readFile(baselinePath) : null;
const baseline = baselineBytes ? JSON.parse(baselineBytes) : null;
if (intervention) {
  assert.equal(process.env.DOPPLER_TEST_ONLY_ARITHMETIC, '1', 'Reference injection requires DOPPLER_TEST_ONLY_ARITHMETIC=1');
  assert(operation === 'recurrent' && !observeRecurrent && !replayPrefixes, 'Interventions use separate diagnostic runs');
}
assert(!observeActivation || operation === 'conv');
assert(!observeRecurrent || operation === 'recurrent');
assert(!replayPrefixes || (operation === 'recurrent' && !observeRecurrent), 'Prefix replay uses the unchanged recurrent shader');
assert(!activationCandidate || operation === 'conv', 'Activation candidate is qualified only for convolution diagnosis');
let shader = observeActivation ? originalShader.replace(
  '    conv_out[token_idx * params.conv_dim + channel] = silu(mixed);',
  `    let z = ${activationCandidate ? 'diagnostic_exp' : 'exp'}(-abs(mixed));
    observed_activation[token_idx * params.conv_dim + channel] = vec4<f32>(mixed, z, 1.0 + z, ${activationCandidate ? 'diagnostic_reciprocal(1.0 + z)' : '1.0 / (1.0 + z)'});
    conv_out[token_idx * params.conv_dim + channel] = silu(mixed);`)
  + '\n@group(0) @binding(5) var<storage, read_write> observed_activation: array<vec4<f32>>;'
  : originalShader;
if (activationCandidate) shader = buildLinearActivationDiagnostic(shader);
if (observeRecurrent) shader = observeRecurrentShader(shader, independent.layout, observedStages);
const injected = intervention ? recurrentInterventionOperands(independent, intervention) : null;
if (intervention) shader = interveneRecurrentShader(shader, injected.layout, intervention);
if (compensated) shader = buildRecurrentAccumulationDiagnostic(shader);
const referenceBytes = injected ? Buffer.from(injected.values.buffer) : null;
const traceBytes = observeRecurrent ? independent.layout.length * 4
  : observeActivation ? input.params.numTokens * input.params.convDim * 16 : 0;
const hash = b => createHash('sha256').update(b).digest('hex');
const { chromium } = createRequire(resolve(reploidRoot, 'package.json'))('playwright');
async function boundedReplay(page, callback, args) {
  let timer;
  try {
    return await Promise.race([page.evaluate(callback, args), new Promise((_, reject) => {
      timer = setTimeout(() => reject(Error('Isolated GPU replay exceeded 60 seconds')), 60000);
    })]);
  } finally { clearTimeout(timer); }
}
const receipt = { scope: 'Identical captured operands through one linear-attention operation', operation,
  captureSha256: hash(bytes), shaderSha256: hash(shader), archiveSha256: capture.archiveSha256,
  sourceSubstitution: observeActivation || observeRecurrent || activationCandidate || Boolean(intervention) || compensated,
  observeActivation, observeRecurrent, activationCandidate, compensated,
  coordinate, traceLayout: observeRecurrent ? independent.layout : null,
  observedStages: observeRecurrent ? observedStages : null,
  replayPrefixes,
  intervention, referenceInjectionSha256: referenceBytes ? hash(referenceBytes) : null,
  interventionScope: intervention ? 'Test-only F32-rounded independent operands; never a runtime or release candidate' : null,
  macStrictMath, strictMathScope: macStrictMath ? 'Chrome developer-only Metal diagnostic; not a production solution' : null,
  originalShaderSha256: hash(originalShader),
  upstreamSourceSubstitution: capture.sourceSubstitution, results: [] };
await writeFile(`${output}.wgsl`, shader);
for (const platform of ['mac', 'linux']) {
  const browser = platform === 'mac'
    ? await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal',
      ...(macStrictMath ? ['--enable-webgpu-developer-features'] : [])] })
    : await chromium.connect(process.env.REPLOID_EXECUTOR_WS);
  const context = await browser.newContext();
  try {
    const page = await context.newPage(); await page.goto('http://localhost:8000/config/chat-files.json');
    if (referenceBytes) {
      const text = referenceBytes.toString('base64');
      await page.evaluate(() => { globalThis.recurrentReferenceChunks = []; });
      // Bound each protocol message; a full per-token state trace can otherwise
      // monopolize the remote browser transport with one very large argument.
      for (let offset = 0; offset < text.length; offset += 262144) {
        await boundedReplay(page, chunk => { globalThis.recurrentReferenceChunks.push(chunk); }, text.slice(offset, offset + 262144));
      }
    }
    const result = await boundedReplay(page, async ({ input, expected, shader, operation, observeRecurrent, replayPrefixes, traceBytes, injectedReference, strictMath }) => {
      const referenceData = injectedReference ? globalThis.recurrentReferenceChunks.join('') : null;
      delete globalThis.recurrentReferenceChunks;
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice({ requiredLimits:
        operation === 'recurrent' ? { maxStorageBuffersPerShaderStage: observeRecurrent || referenceData ? 10 : 9 } : {} });
      const owned = [], p = input.params;
      const buffer = (size, usage) => { const b = device.createBuffer({ size, usage }); owned.push(b); return b; };
      try {
        let strictMathRead = false;
        const descriptor = { code: shader };
        if (strictMath) Object.defineProperty(descriptor, 'strictMath', { get() { strictMathRead = true; return true; } });
        const module = device.createShaderModule(descriptor);
        if (strictMath && !strictMathRead) throw Error('Browser did not consume the requested developer-only strictMath option');
        const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
        if (errors.length) throw Error(errors.map(m => m.message).join('\n'));
        // Interventions can remove resource reads; retain the original binding
        // numbering explicitly instead of relying on an optimized auto layout.
        const pipelineLayout = referenceData ? device.createPipelineLayout({ bindGroupLayouts: [
          device.createBindGroupLayout({ entries: Array.from({ length: 11 }, (_, binding) => ({ binding,
            visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 0 ? 'uniform'
              : binding === 8 || binding === 9 ? 'storage' : 'read-only-storage' } })) })] }) : 'auto';
        const pipeline = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: {
          module, entryPoint: 'main', constants: { WORKGROUP_SIZE: operation === 'conv' ? 256 : 128 } } });
        const uniforms = new ArrayBuffer(64), view = new DataView(uniforms);
        ['numTokens', 'convDim', 'convKernelSize', 'numVHeads', 'numKHeads', 'headKDim',
          'headVDim', 'qSize', 'kSize', 'valueDim', 'qRep'].forEach((key, i) => view.setUint32(i * 4, p[key], true));
        view.setUint32(44, p.normMode === 'per_head' ? 1 : 0, true);
        view.setFloat32(48, p.rmsNormEps, true); view.setFloat32(52, p.qkL2NormEps, true);
        view.setUint32(56, Number(p.abPacked) | Number(p.qkvzPacked) << 1, true);
        view.setUint32(60, p.bProjOffsetElements, true);
        const uniform = buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(uniform, 0, uniforms);
        const outputRole = operation === 'conv' ? 'convOutput' : 'coreOutput';
        const roles = operation === 'conv' ? ['qkv', 'convWeight', 'convState', outputRole]
          : ['convOutput', 'z', 'a', 'b', 'dtBias', 'aLog', 'normWeight', 'recurrentState', outputRole];
        const buffers = roles.map(role => {
          const tensor = (role === 'convOutput' || role === 'coreOutput' ? expected : input).tensors.find(t => t.role === role);
          if (!tensor) throw Error(`Missing captured ${role}`);
          const b = buffer(tensor.bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
          if (role !== outputRole) device.queue.writeBuffer(b, 0, Uint8Array.from(atob(tensor.data), c => c.charCodeAt(0)));
          return b;
        });
        const size = p.numTokens * (operation === 'conv' ? p.convDim : p.valueDim) * 4;
        const trace = traceBytes ? buffer(traceBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : null;
        const traceReadback = traceBytes ? buffer(traceBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) : null;
        const stateBytes = operation === 'recurrent' ? p.numVHeads * p.headKDim * p.headVDim * 4 : 0;
        const stateReadback = stateBytes ? buffer(stateBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ) : null;
        let referenceBuffer = null;
        if (referenceData) {
          const decoded = atob(referenceData), data = new Uint8Array(decoded.length);
          for (let i = 0; i < decoded.length; i++) data[i] = decoded.charCodeAt(i);
          referenceBuffer = buffer(data.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(referenceBuffer, 0, data);
        }
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
          [uniform, ...buffers, ...(trace ? [trace] : []), ...(referenceBuffer ? [referenceBuffer] : [])]
            .map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const staging = buffer(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(operation === 'conv' ? Math.ceil(p.convDim / 256) : p.numVHeads); pass.end();
        encoder.copyBufferToBuffer(buffers.at(-1), 0, staging, 0, size);
        if (trace) encoder.copyBufferToBuffer(trace, 0, traceReadback, 0, traceBytes);
        if (stateReadback) encoder.copyBufferToBuffer(buffers[roles.indexOf('recurrentState')], 0, stateReadback, 0, stateBytes);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Uint8Array(staging.getMappedRange().slice(0)); staging.unmap();
        let text = '';
        for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
        let activationData = null;
        if (traceReadback) {
          await traceReadback.mapAsync(GPUMapMode.READ);
          const data = new Uint8Array(traceReadback.getMappedRange().slice(0)); traceReadback.unmap();
          let text = '';
          for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
          activationData = btoa(text);
        }
        let stateData = null;
        if (stateReadback) {
          await stateReadback.mapAsync(GPUMapMode.READ);
          const data = new Uint8Array(stateReadback.getMappedRange().slice(0)); stateReadback.unmap();
          let text = '';
          for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
          stateData = btoa(text);
        }
        const prefixes = [];
        if (replayPrefixes) {
          const initial = input.tensors.filter(t => t.role === 'recurrentState');
          if (initial.length !== 1) throw Error('Prefix replay requires one captured initial state');
          const initialBytes = Uint8Array.from(atob(initial[0].data), c => c.charCodeAt(0));
          const readBase64 = async (staging, offset, size) => {
            await staging.mapAsync(GPUMapMode.READ);
            const data = new Uint8Array(staging.getMappedRange().slice(offset, offset + size)); staging.unmap();
            let text = '';
            for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
            return btoa(text);
          };
          for (let count = 1; count <= p.numTokens; count++) {
            // Every prefix starts from identical captured state. No shader edits,
            // injected reference values, or inherited state from the prior replay.
            device.queue.writeBuffer(buffers[roles.indexOf('recurrentState')], 0, initialBytes);
            device.queue.writeBuffer(uniform, 0, new Uint32Array([count]));
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(p.numVHeads); pass.end();
            encoder.copyBufferToBuffer(buffers.at(-1), 0, staging, 0, count * p.valueDim * 4);
            encoder.copyBufferToBuffer(buffers[roles.indexOf('recurrentState')], 0, stateReadback, 0, stateBytes);
            device.queue.submit([encoder.finish()]);
            prefixes.push({ token: count - 1,
              output: await readBase64(staging, (count - 1) * p.valueDim * 4, p.valueDim * 4),
              state: await readBase64(stateReadback, 0, stateBytes) });
          }
        }
        return { data: btoa(text), activationData, stateData, strictMathRequested: strictMath, strictMathRead,
          prefixes, vendor: adapter.info.vendor, architecture: adapter.info.architecture };
      } finally { for (const b of owned) b.destroy(); device.destroy(); }
    }, { input, expected, shader, operation, observeRecurrent, replayPrefixes, traceBytes,
      injectedReference: Boolean(referenceBytes), strictMath: macStrictMath && platform === 'mac' });
    receipt.results.push({ platform, browser: browser.version(), ...result });
    console.log(JSON.stringify({ platform, operation, completed: true }));
  } catch (error) {
    receipt.failure = { platform, message: error.message };
    await writeFile(`${output}.failure.json`, JSON.stringify(receipt));
    console.error(`Replay failed on ${platform}: ${error.message}`);
    throw error;
  } finally { await context.close(); await browser.close(); }
}
const values = text => { const b = Buffer.from(text, 'base64'); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const compare = (a, b) => {
  assert.equal(a.length, b.length); let maxDifference = 0, differentElements = 0, squaredError = 0;
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]) && Number.isFinite(b[i]));
    maxDifference = Math.max(maxDifference, Math.abs(a[i] - b[i])); differentElements += a[i] !== b[i];
    squaredError += (a[i] - b[i]) ** 2;
  }
  return { elements: a.length, differentElements, maxDifference, rmsError: Math.sqrt(squaredError / a.length) };
};
const actual = receipt.results.map(r => values(r.data));
receipt.crossPlatform = compare(...actual);
if (baseline) {
  assert.equal(baseline.captureSha256, receipt.captureSha256, 'Baseline operands changed');
  assert.deepEqual(baseline.coordinate, coordinate, 'Baseline capture coordinates changed');
  assert.equal(baseline.sourceSubstitution, false, 'Baseline must use unmodified source');
  assert.equal(baseline.originalShaderSha256, receipt.originalShaderSha256, 'Baseline shader changed');
  receipt.observationValidation = { baselineSha256: hash(baselineBytes),
    comparisons: receipt.results.map(r => {
      const matches = baseline.results.filter(b => b.platform === r.platform);
      assert.equal(matches.length, 1); const original = matches[0];
      assert.equal(original.browser, r.browser); assert.equal(original.vendor, r.vendor);
      assert.equal(original.architecture, r.architecture);
      return { platform: r.platform, output: compare(values(r.data), values(original.data)),
        state: compare(values(r.stateData), values(original.stateData)) };
    }) };
  receipt.observationValidation.preservesUninstrumentedExecution = receipt.observationValidation.comparisons
    .every(c => c.output.differentElements === 0 && c.state.differentElements === 0);
}
if (observeActivation) {
  const traces = receipt.results.map(r => values(r.activationData));
  receipt.activationComparison = ['mixed', 'exponential', 'denominator', 'reciprocal'].map((stage, lane) => ({ stage,
    ...compare(...traces.map(t => t.filter((_, i) => i % 4 === lane))) }));
}
const expectedValues = values(expected.tensors.find(t => t.role === (operation === 'conv' ? 'convOutput' : 'coreOutput')).data);
receipt.macCaptureAgreement = compare(actual[0], expectedValues.subarray(0, actual[0].length));
if (independent) {
  const states = receipt.results.map(r => values(r.stateData));
  receipt.stateComparison = compare(...states);
  receipt.macStateCaptureAgreement = compare(states[0], decodeCapturedTensor(expected, 'recurrentState').subarray(0, states[0].length));
  receipt.reference = { ...RECURRENT_REFERENCE, scope: 'Independent Float64 equations on identical captured F32 operands',
    traceSha256: hash(Buffer.from(independent.trace.buffer)),
    comparisons: actual.map((v, i) => ({ platform: receipt.results[i].platform,
      output: compare(v, independent.output), state: compare(states[i], independent.finalState) })) };
  if (replayPrefixes) {
    const f = independent.layout.fields.updatedState;
    const stateSize = independent.finalState.length, width = input.params.valueDim;
    receipt.prefixComparisons = Array.from({ length: input.params.numTokens }, (_, token) => {
      const observations = receipt.results.map(r => r.prefixes[token]);
      const prefixStates = observations.map(p => values(p.state));
      const prefixOutput = observations.map(p => values(p.output));
      return { token, crossPlatform: { output: compare(...prefixOutput), state: compare(...prefixStates) },
        reference: observations.map((p, i) => ({ platform: receipt.results[i].platform,
          output: compare(prefixOutput[i], independent.output.subarray(token * width, (token + 1) * width)),
          state: compare(prefixStates[i], independent.trace.subarray(f.offset + token * stateSize, f.offset + (token + 1) * stateSize)),
          unchangedOutput: compare(prefixOutput[i], actual[i].subarray(token * width, (token + 1) * width)) })) };
    });
    receipt.prefixResetVerified = receipt.results.every((r, i) => r.prefixes.at(-1).state === r.stateData &&
      receipt.prefixComparisons.every(p => p.reference[i].unchangedOutput.differentElements === 0));
    assert(receipt.prefixResetVerified, 'Prefix replay changed original state/output');
  }
  if (observeRecurrent) {
    const traces = receipt.results.map(r => values(r.activationData));
    receipt.recurrentStages = Object.entries(independent.layout.fields).filter(([stage]) => observedStages.includes(stage)).map(([stage, f]) => {
      const reference = independent.trace.subarray(f.offset, f.offset + f.length);
      const observed = traces.map(t => t.subarray(f.offset, f.offset + f.length));
      const perToken = f.length / input.params.numTokens;
      return { stage, crossPlatform: compare(...observed),
        reference: observed.map((v, i) => ({ platform: receipt.results[i].platform, ...compare(v, reference) })),
        tokens: Array.from({ length: input.params.numTokens }, (_, token) => {
          const slices = observed.map(t => t.subarray(token * perToken, (token + 1) * perToken));
          const ref = reference.subarray(token * perToken, (token + 1) * perToken);
          return { token, crossPlatform: compare(...slices),
            reference: slices.map((v, i) => ({ platform: receipt.results[i].platform, ...compare(v, ref) })) };
        }) };
    });
  }
}
if (operation === 'conv') {
  const p = input.params, read = role => values(input.tensors.find(t => t.role === role).data);
  const qkv = read('qkv'), weights = read('convWeight'), state = Float64Array.from(read('convState'));
  const reference = new Float64Array(p.numTokens * p.convDim);
  for (let t = 0; t < p.numTokens; t++) for (let c = 0; c < p.convDim; c++) {
    const base = c * p.convKernelSize;
    for (let k = 0; k < p.convKernelSize - 1; k++) state[base + k] = state[base + k + 1];
    state[base + p.convKernelSize - 1] = qkv[t * (p.convDim + (p.qkvzPacked ? p.valueDim : 0)) + c];
    let mixed = 0;
    for (let k = 0; k < p.convKernelSize; k++) mixed += state[base + k] * weights[base + k];
    reference[t * p.convDim + c] = mixed / (1 + Math.exp(-mixed));
  }
  receipt.reference = { scope: 'Independent Float64 convolution and SiLU; never fed into inference',
    sha256: hash(Buffer.from(reference.buffer)), comparisons: actual.map((v, i) => ({ platform: receipt.results[i].platform,
      ...compare(v, reference), rmsError: Math.sqrt(v.reduce((sum, x, j) => sum + (x - reference[j]) ** 2, 0) / v.length) })) };
}
await writeFile(output, JSON.stringify(receipt));
if (observeRecurrent && !receipt.observationValidation.preservesUninstrumentedExecution) {
  console.error('Intermediate probe changed execution; retain as a failed observer, not a production trace');
  process.exitCode = 2;
}
console.log(JSON.stringify({ crossPlatform: receipt.crossPlatform, macCaptureAgreement: receipt.macCaptureAgreement,
  stateComparison: receipt.stateComparison, macStateCaptureAgreement: receipt.macStateCaptureAgreement,
  reference: receipt.reference, activationComparison: receipt.activationComparison,
  recurrentStages: receipt.recurrentStages?.map(({ tokens, reference, ...stage }) => stage) }));
