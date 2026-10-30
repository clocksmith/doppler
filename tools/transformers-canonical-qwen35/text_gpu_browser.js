import * as ort from '/ort/ort.webgpu.min.mjs';
import { createTjsTextModel } from '/adapter/tjs_model.js';
import { teacherCases } from '/adapter/teacher_cases.js';
import { ObservedSession } from '/adapter/observed_session.js';
import { uploadGpuTensor } from '/adapter/observe_gpu.js';

const state = { schema: 'doppler.canonical-tjs-gpu-diagnostic.v1', qualification: false,
  scope: 'Untimed TJS public-forward diagnostic; strict CPU state gate remains failed; no speed or qualification claim',
  phase: 'initializing', executionCompleted: false, checks: [], gpuErrors: [], events: [] };

async function phase(name) {
  state.phase = name;
  state.events.push(name);
  await fetch('/event', { method: 'POST', body: JSON.stringify({ phase: name }) });
}

async function main() {
  const descriptor = await (await fetch('/probe.json')).json();
  const { config, modelReceipt, cpuReceipt, teacher, modelConfig, policy, sessions: sessionDescriptions } = descriptor;
  if (config.diagnosticOnly !== true) throw new Error('This runner is authorized only for explicitly labeled diagnostic work');
  state.cpuGate = descriptor.cpuGate;
  state.canonicalInputs = modelReceipt.inputs;
  state.physicalKV = modelReceipt.contract.kvPhysicalDtype;
  state.weightDeviceReadback = modelReceipt.readbackPorts ? 'required from actual inference sessions' : 'not-run; frozen ONNX initializer binding is separate evidence';
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || !adapter.info.vendor?.toLowerCase().includes(config.requiredVendor.toLowerCase())) {
    throw new Error('Missing or unexpected physical GPU adapter');
  }
  const requiredLimits = Object.fromEntries(config.requiredLimitNames.map(name => [name, adapter.limits[name]]));
  if (Object.values(requiredLimits).some(value => !Number.isSafeInteger(value))) throw new Error('Requested adapter limit is unavailable');
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures: config.requiredFeatures });
  let executionDevice = device;
  try {
  device.addEventListener('uncapturederror', event => state.gpuErrors.push(event.error.message ?? String(event.error)));
  device.lost.then(info => { if (info.reason !== 'destroyed') state.gpuErrors.push(`Device lost: ${info.message}`); });
  state.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description };
  state.requestedLimits = requiredLimits;
  state.deviceLimits = { maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize };
  ort.env.webgpu.device = device;
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'verbose';
  // The browser import map resolves TJS's public ORT dependency to this same
  // installed module. The library selects its ordinary browser backend.
  const tjs = await import('/tjs/transformers.web.min.js');
  const raw = {};
  let model, decoder, cache;
  if (config.graphOptimizationLevel !== 'disabled') throw new Error('Diagnostic must execute the same unoptimized graph as the CPU reference');
  const options = { executionProviders: ['webgpu'], graphOptimizationLevel: config.graphOptimizationLevel,
    extra: { session: { disable_cpu_ep_fallback: '1' } } };
  state.sessionOptions = options;
  state.sessionInitializerBytes = descriptor.sessionInitializerBytes;
  state.sessionResidency = { sessions: Object.keys(sessionDescriptions),
    scope: 'Independent sessions; learned weights may be duplicated in GPU residency' };
  try {
    for (const [name, graph] of Object.entries(sessionDescriptions)) {
      await phase(`create-session-${name}`);
      const preferredOutputLocation = name === 'embedding' ? { inputs_embeds: 'gpu-buffer' } :
        Object.fromEntries(modelReceipt.cacheBindings.map(port => [port.output, 'gpu-buffer']));
      const graphName = graph.graphName;
      for (const port of modelReceipt.readbackPorts?.[graphName] ?? []) preferredOutputLocation[port.output] = 'gpu-buffer';
      raw[name] = await ort.InferenceSession.create(graph.url, { ...options, preferredOutputLocation,
        externalData: graph.externalData.map(file => ({ path: file.path, data: file.url })) });
      const actualDevice = await ort.env.webgpu.device;
      if (!actualDevice?.queue || !actualDevice.adapterInfo?.vendor?.toLowerCase().includes(config.requiredVendor.toLowerCase())) {
        throw new Error('ORT actual execution device is missing or uses the wrong physical adapter');
      }
      if (actualDevice !== executionDevice) {
        executionDevice = actualDevice;
        executionDevice.addEventListener('uncapturederror', event => state.gpuErrors.push(event.error.message ?? String(event.error)));
        executionDevice.lost.then(info => { if (info.reason !== 'destroyed') state.gpuErrors.push(`ORT device lost: ${info.message}`); });
      }
      state.requestedDeviceHonored = actualDevice === device;
      state.adapter = { vendor: actualDevice.adapterInfo.vendor, architecture: actualDevice.adapterInfo.architecture,
        device: actualDevice.adapterInfo.device, description: actualDevice.adapterInfo.description };
      state.deviceLimits = { maxBufferSize: actualDevice.limits.maxBufferSize,
        maxStorageBufferBindingSize: actualDevice.limits.maxStorageBufferBindingSize };
      raw[name].config = { dtype: 'fp32', device: 'webgpu' };
    }
    const modelSessions = { ...raw };
    if (modelReceipt.readbackPorts) {
      state.weightReadbacks = [];
      for (const [name, session] of Object.entries(sessionDescriptions)) {
        modelSessions[name] = new ObservedSession(raw[name], modelReceipt.readbackPorts[session.graphName], async (port, tensor) => {
            await phase(`observe-initializer-${name}-${port.output}`);
            const receipt = await uploadGpuTensor(executionDevice, tensor, port.bytes, config.weightReadbackChunkBytes,
              `/weight-readback/${encodeURIComponent(name + ':' + port.output)}`, config.observationStepTimeoutMs);
            if (!receipt.pass) throw new Error(`Actual GPU initializer bytes differ: ${name} ${port.initializer}`);
            state.weightReadbacks.push({ session: name, output: port.output, source: port.source,
              location: tensor.location, type: tensor.type, bytes: port.bytes, pass: true });
        });
      }
    }
    ({ model, decoder } = createTjsTextModel(tjs, modelConfig, modelSessions, policy));
    state.implementation = `Transformers.js 4.2 public Qwen3_5ForCausalLM.forward + explicit session routing; physical KV ${state.physicalKV}`;
    state.prompts = [];
    const cases = teacher ? teacherCases(teacher, state.prompts) : cpuReceipt.cases.values();
    let nextCase = cases.next();
    while (!nextCase.done) {
      const test = nextCase.value;
      await phase(`forward-${test.name}`);
      if (test.positionOffset === 0) {
        await cache?.dispose();
        cache = new tjs.DynamicCache();
      }
      const count = test.inputIds.length;
      const input = new tjs.Tensor('int64', BigInt64Array.from(test.inputIds, BigInt), [1, count]);
      const indices = (type, values, shape) => new tjs.Tensor(type,
        type === 'int32' ? Int32Array.from(values) : BigInt64Array.from(values, BigInt), shape);
      const mask = indices(decoder.integerInputTypes.attention_mask, new Array(test.positionOffset + count).fill(1),
        [1, test.positionOffset + count]);
      const positions = indices(decoder.integerInputTypes.position_ids, Array.from({ length: 3 * count },
        (_, index) => test.positionOffset + index % count), [3, 1, count]);
      let output, embedding, adopted = false, sampledToken;
      try {
        embedding = await model.encode_text({ input_ids: input });
        if (embedding.location !== 'gpu-buffer' || embedding.type !== 'float32') {
          throw new Error('TJS embedding output must remain F32 GPU resident');
        }
        output = await model.forward({ input_ids: input, inputs_embeds: embedding, attention_mask: mask,
          position_ids: positions, past_key_values: cache });
        const actual = output.logits.data;
        const expectedDescriptor = test.logits.find(value => value.engine === 'native');
        let expected;
        if (expectedDescriptor) {
          const binary = await (await fetch(`/native/${encodeURIComponent(expectedDescriptor.file)}`)).arrayBuffer();
          const checksum = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', binary)),
            value => value.toString(16).padStart(2, '0')).join('');
          if (checksum !== expectedDescriptor.sha256) throw new Error('Native reference logit bytes differ');
          expected = new Float32Array(binary);
        }
        if ((expected && actual.length !== expected.length) || actual.length !== modelConfig.text_config.vocab_size) {
          throw new Error('Full vocabulary logit shape mismatch');
        }
        let maximum = 0, sum = 0, failures = 0, argmax = 0, referenceArgmax = 0;
        for (let index = 0; index < actual.length; index++) {
          const error = expected ? Math.abs(actual[index] - expected[index]) : 0;
          if (!Number.isFinite(actual[index]) || error > modelReceipt.contract.logitAbsoluteTolerance) failures++;
          maximum = Math.max(maximum, error);
          sum += error;
          if (actual[index] > actual[argmax]) argmax = index;
          if (expected && expected[index] > expected[referenceArgmax]) referenceArgmax = index;
        }
        sampledToken = argmax;
        const expectedToken = teacher ? test.expectedTokenId : referenceArgmax;
        state.checks.push({ case: test.name, elements: actual.length, referenceSnapshot: Boolean(expected),
          maxAbsoluteError: expected ? maximum : null, meanAbsoluteError: expected ? sum / actual.length : null,
          failedLogits: expected ? failures : null,
          logitsPass: expected ? failures === 0 && test.teacherPrefixAligned !== false : failures === 0 ? null : false,
          teacherPrefixAligned: test.teacherPrefixAligned ?? null,
          argmax, nativeArgmax: expectedToken, exactArgmax: argmax === expectedToken,
          states: decoder.ports.map(port => ({ name: port.output, type: output[port.output].type,
            location: output[port.output].location, shape: output[port.output].dims })) });
        if (config.captureStateSnapshots === true && teacher.contract.snapshotSteps.includes(test.step)) {
          for (const port of decoder.ports) {
            const tensor = output[port.output];
            const bytes = 4 * tensor.dims.reduce((a, b) => a * b, 1);
            await uploadGpuTensor(executionDevice, tensor, bytes, config.weightReadbackChunkBytes,
              `/state-readback/${encodeURIComponent(test.promptId + ':' + test.step + ':' + port.output)}`, config.observationStepTimeoutMs);
          }
        }
        cache.update(Object.fromEntries(decoder.ports.map(port => [port.input, output[port.output]])));
        adopted = true;
      } finally {
        embedding?.dispose();
        output?.logits?.dispose();
        if (output && !adopted) for (const port of decoder.ports) output[port.output]?.dispose();
        input.dispose(); mask.dispose(); positions.dispose();
      }
      nextCase = cases.next(sampledToken);
    }
    await executionDevice.queue.onSubmittedWorkDone();
    if (state.gpuErrors.length) throw new Error(state.gpuErrors.join('\n'));
    state.paddingAndScheduler = decoder.work;
    state.logitsPass = state.checks.every(check => check.logitsPass !== false && check.exactArgmax);
    if (teacher) state.teacherOutputGate = state.logitsPass && state.prompts.length === 4 &&
      state.prompts.every(prompt => prompt.completedBudget && prompt.exactTokens && prompt.endedByEOS === prompt.expectedEndedByEOS);
    state.executionCompleted = true;
    await phase('completed-diagnostic');
  } finally {
    await cache?.dispose();
    if (model) await model.dispose();
    else for (const session of Object.values(raw)) await session.release();
  }
  } finally { if (executionDevice !== device) executionDevice.destroy(); device.destroy(); }
}

try { await main(); }
catch (error) { state.error = String(error); state.stack = error.stack; }
globalThis.RESULT = state;
