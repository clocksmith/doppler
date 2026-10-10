import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

/** Read-only copies before full attention and after its output projection input.
 * Shares the fixture's debugger session so independent pause handlers cannot race.
 * Captures full prefill tensors; never changes arithmetic, precision or cache data. */
export async function observeAttentionCache(context, packageRoot, captures, options = {}) {
  let locations = [
    { file: 'inference/pipelines/text/attention/interpreter/recorded.js',
      marker: '  let attnOutput = null;', condition: 'options.numTokens > 1 && [3, 7, 11].includes(options.layerIdx)',
      expression: 'captureAttentionCache("inputs", options)' },
    { file: 'inference/pipelines/text/attention/interpreter.js',
      marker: '  let residualFused = false;', condition: 'numTokens > 1 && [3, 7, 11].includes(layerIdx)',
      expression: 'captureAttentionCache("outputs", {recorder, layerIdx, numTokens, numHeads, numKVHeads, headDim, attnOutput, attnForProjection})' },
    { file: 'gpu/kernels/matmul.js', marker: "    const tensor = createTensor(C, actualOutputDtype, [M, N], 'matmul_output');",
      condition: 'options.layerIdx === 0 && ["linear_qkv_proj", "linear_qkvz_proj"].includes(options.role)',
      expression: 'captureProjection({recorder, M, N, K, alpha, variant, pathVariant, constants, config, layerIdx:options.layerIdx, role:options.role, input:matmulInput, weight:bBuffer, output:C, residual:options.residualTensor, aOffset, bOffset, cOffset, bindingSizes, cBindingSize, dtype:actualOutputDtype})' },
    { file: 'gpu/kernels/linear-attention-core.js', marker: '      const recurrentBindGroup = device.createBindGroup({',
      condition: 'options.layerIdx === 0 && numTokens > 1',
      expression: 'captureLinearCore("inputs", {recorder, numTokens, layerState, options, qkvTensor, zTensor, aTensor, bTensor})' },
    { file: 'gpu/kernels/linear-attention-core.js', marker: '      recorder.trackTemporaryBuffer(convOutBuffer);',
      condition: 'options.layerIdx === 0 && numTokens > 1',
      expression: 'captureLinearCore("outputs", {recorder, numTokens, layerState, options, convOutBuffer, outputBuffer})' },
    { file: 'gpu/kernels/linear-attention-core.js', marker: "    const encoder = device.createCommandEncoder({ label: 'linear_attention_core' });",
      condition: 'options.layerIdx === 0 && numTokens > 1',
      expression: 'captureLinearCore("inputs", {device, numTokens, layerState, options, qkvTensor, zTensor, aTensor, bTensor})' },
    { file: 'gpu/kernels/linear-attention-core.js', marker: '    submitted = true;',
      condition: 'options.layerIdx === 0 && numTokens > 1',
      expression: 'captureLinearCore("outputs", {device, numTokens, layerState, options, convOutBuffer, outputBuffer})' },
  ];
  const projectionLocation = locations.find(location => location.expression.startsWith('captureProjection('));
  if (options.attentionLayer !== undefined) {
    assert(options.linearLayer === undefined, 'Select one attention observer');
    assert(Number.isInteger(options.attentionLayer) && options.attentionLayer >= 0);
    locations = locations.filter(location => location.expression.startsWith('captureAttentionCache('))
      .map(location => ({ ...location, condition: location.expression.includes('"inputs"')
        ? `options.layerIdx === ${options.attentionLayer}` : `layerIdx === ${options.attentionLayer}` }));
  }
  if (options.linearLayer !== undefined) {
    assert(Number.isInteger(options.linearLayer) && options.linearLayer >= 0);
    const condition = `options.layerIdx === ${options.linearLayer}`;
    const common = { file: 'gpu/kernels/linear-attention-core.js', condition };
    locations = [
      { ...common, marker: '  if (useFusedDecodeCore) {',
        expression: 'captureLinearCore("inputs", {recorder, device, numTokens, layerState, options, qkvTensor, zTensor, aTensor, bTensor})' },
      ...[
        ['        const output = createTensor(', 0],
        ['      const output = createTensor(', 0],
        ['      const output = createTensor(', 1],
        ['    const output = createTensor(', 0],
      ].map(([marker, occurrence]) => ({ ...common, marker, occurrence,
        expression: 'captureLinearCore("outputs", {recorder, device, numTokens, layerState, options, convOutBuffer, outputBuffer})' })),
    ];
  }
  if (options.linearOnly === true) locations = locations.filter(location => location.expression.startsWith('captureLinearCore('));
  if (options.projectionLayer !== undefined) {
    assert.equal(options.projectionLayer, options.attentionLayer,
      'Projection observation requires its selected attention boundary');
    locations.push({ ...projectionLocation,
      condition: `options.layerIdx === ${options.projectionLayer} && options.role === "o_proj"` });
  }
  if (options.captureCondition) {
    locations = locations.map(location => ({ ...location,
      condition: `(${location.condition}) && (${options.captureCondition})` }));
  }
  for (const location of locations) {
    const lines = (await readFile(resolve(packageRoot, 'src', location.file), 'utf8')).split('\n');
    location.line = lines.flatMap((line, index) => line === location.marker ? [index] : [])[location.occurrence ?? 0] ?? -1;
    assert(location.line >= 0, `Missing observation boundary: ${location.file}`);
  }
  context.on('page', page => {
    page.on('pageerror', error => console.error(`Attention observer page error: ${error.message}`));
    page.on('console', message => {
      if (message.type() === 'error' || message.type() === 'warning') console.error(message.text());
    });
  });
  await context.addInitScript(() => {
    globalThis.attentionCacheObservation = { records: [], errors: [] };
    globalThis.attentionCacheObservationPending = [];
    const pendingLinear = new Map();
    const linearOrdinals = new Map();
    globalThis.captureLinearCore = (boundary, options) => {
      const { recorder, numTokens, layerState: state } = options;
      const device = recorder?.device ?? options.device;
      const encoder = recorder?.getEncoder() ?? device.createCommandEncoder({ label: 'linear_observation_copy' });
      const readbacks = [];
      const params = { numTokens };
      for (const key of ['convDim', 'convKernelSize', 'numVHeads', 'numKHeads', 'headKDim',
        'headVDim', 'qSize', 'kSize', 'valueDim', 'qRep', 'normMode', 'rmsNormEps']) params[key] = state[key];
      for (const key of ['qkL2NormEps', 'abPacked', 'qkvzPacked', 'bProjOffsetElements']) params[key] = options.options[key];
      const coordinate = { layerIdx: options.options.layerIdx,
        step: globalThis.numericalObservation?.step,
        prompt: globalThis.numericalObservation?.prompt };
      if (!Number.isInteger(coordinate.layerIdx) || coordinate.step === undefined || coordinate.prompt === undefined) {
        throw Error('Linear observation requires layer, prompt and step coordinates');
      }
      const key = JSON.stringify(coordinate);
      if (boundary === 'inputs') {
        if (pendingLinear.has(key)) throw Error('Unpaired linear input observation');
        const ordinal = linearOrdinals.get(key) ?? 0;
        linearOrdinals.set(key, ordinal + 1);
        pendingLinear.set(key, ordinal);
      }
      if (!pendingLinear.has(key)) throw Error('Linear output has no matching input');
      const record = { boundary: `linear-${boundary}`, ...coordinate,
        dispatch: pendingLinear.get(key), params, tensors: [] };
      if (boundary === 'outputs') pendingLinear.delete(key);
      attentionCacheObservation.records.push(record);
      const buffers = boundary === 'inputs'
        ? [['qkv', options.qkvTensor.buffer], ['z', options.zTensor.buffer],
          ['a', options.aTensor.buffer], ['b', options.bTensor.buffer],
          ...['convWeight', 'convState', 'dtBias', 'aLog', 'normWeight', 'recurrentState']
            .map(role => [role, state[role + 'GPU']])]
        : [['convOutput', options.convOutBuffer], ['coreOutput', options.outputBuffer],
          ['convState', state.convStateGPU], ['recurrentState', state.recurrentStateGPU]];
      for (const [role, buffer] of buffers) {
        if (role === 'convOutput' && !buffer) continue; // Fused decode has no intermediate convolution buffer.
        const bytes = buffer.size;
        const item = { role, bytes, data: null }; record.tensors.push(item);
        const staging = device.createBuffer({ size: bytes,
          label: 'linear_core_observation', usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
        const readback = async () => {
          let mapped = false;
          try {
            await staging.mapAsync(GPUMapMode.READ); mapped = true;
            const data = new Uint8Array(staging.getMappedRange().slice(0));
            let text = '';
            for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
            item.data = btoa(text);
          } catch (error) { attentionCacheObservation.errors.push(error.message); }
          finally { if (mapped) staging.unmap(); staging.destroy(); }
        };
        if (recorder) recorder.enqueueCompletionTask(readback);
        else readbacks.push(readback);
      }
      if (!recorder) {
        device.queue.submit([encoder.finish()]);
        attentionCacheObservationPending.push(Promise.all(readbacks.map(readback => readback())));
      }
    };
    globalThis.captureProjection = options => {
      const { recorder, M, N, K, alpha, variant, pathVariant, constants, config } = options;
      const record = { boundary: 'projection', phase: M === 1 ? 'decode' : 'prefill',
        layerIdx: options.layerIdx, role: options.role,
        step: globalThis.numericalObservation?.step, prompt: globalThis.numericalObservation?.prompt,
        M, N, K, alpha, variant, pathVariant, constants, config, tensors: [] };
      attentionCacheObservation.records.push(record);
      for (const [role, buffer, offset, bytes] of [
        ['input', options.input.buffer, options.aOffset, options.bindingSizes.aBindingSize],
        ['weight', options.weight, options.bOffset, options.bindingSizes.bBindingSize],
        ['output', options.output, options.cOffset, options.cBindingSize],
        ...(options.residual ? [['residual', options.residual.buffer, 0, options.residual.buffer.size]] : []),
      ]) {
        const item = { role, bytes, offset, data: null }; record.tensors.push(item);
        const staging = recorder.device.createBuffer({ size: bytes, label: 'projection_observation',
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        recorder.getEncoder().copyBufferToBuffer(buffer, offset, staging, 0, bytes);
        recorder.enqueueCompletionTask(async () => {
          let mapped = false;
          try {
            await staging.mapAsync(GPUMapMode.READ); mapped = true;
            const bytes = new Uint8Array(staging.getMappedRange().slice(0));
            let text = '';
            for (let i = 0; i < bytes.length; i += 16384) text += String.fromCharCode(...bytes.subarray(i, i + 16384));
            item.data = btoa(text);
          } catch (error) { attentionCacheObservation.errors.push(error.message); }
          finally { if (mapped) staging.unmap(); staging.destroy(); }
        });
      }
    };
    globalThis.captureAttentionCache = (boundary, options) => {
      const { recorder, layerIdx, numTokens, numHeads, numKVHeads, headDim } = options;
      const observation = attentionCacheObservation;
      const count = numTokens * numKVHeads * headDim;
      const cachedCount = (options.kvState?.kvLenForAttention ?? numTokens) * numKVHeads * headDim;
      const tensors = boundary === 'inputs'
        ? [['q', options.qTensor, numTokens * numHeads * headDim],
          ['k', options.kTensor, count], ['v', options.vTensor, count],
          ['cachedK', options.cachedKTensor, cachedCount], ['cachedV', options.cachedVTensor, cachedCount],
          ['gate', options.qGateTensor, numTokens * numHeads * headDim]]
        : [['core', options.attnOutput, numTokens * numHeads * headDim],
          ['projectionInput', options.attnForProjection, numTokens * numHeads * headDim]];
      const record = { boundary, layerIdx, numTokens, numHeads, numKVHeads, headDim,
        step: globalThis.numericalObservation?.step, prompt: globalThis.numericalObservation?.prompt,
        scale: options.attnScale, variant: options.attentionKernelVariant,
        plan: options.attentionPlan, tensors: [] };
      observation.records.push(record);
      for (const [role, tensor, elements] of tensors) {
        if (!tensor) continue;
        const bytes = elements * (tensor.dtype === 'f16' ? 2 : 4);
        if (!['f16', 'f32'].includes(tensor.dtype) || !Number.isSafeInteger(bytes) || bytes > tensor.buffer.size) {
          throw Error(`Invalid ${role} observation geometry`);
        }
        const item = { role, dtype: tensor.dtype, shape: tensor.shape, elements, bytes, data: null };
        record.tensors.push(item);
        const staging = recorder.device.createBuffer({ size: bytes,
          label: 'attention_cache_observation', usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        recorder.getEncoder().copyBufferToBuffer(tensor.buffer, 0, staging, 0, bytes);
        recorder.enqueueCompletionTask(async () => {
          let mapped = false;
          try {
            await staging.mapAsync(GPUMapMode.READ); mapped = true;
            const data = new Uint8Array(staging.getMappedRange().slice(0));
            let text = '';
            for (let i = 0; i < data.length; i += 16384) text += String.fromCharCode(...data.subarray(i, i + 16384));
            item.data = btoa(text);
          } catch (error) { observation.errors.push(error.message); }
          finally { if (mapped) staging.unmap(); staging.destroy(); }
        });
      }
    };
  });
  const createSession = context.newCDPSession.bind(context);
  context.newCDPSession = async page => {
    const session = await createSession(page);
    await session.send('Debugger.enable');
    const expressions = new Map();
    for (const location of locations) {
      const breakpoint = await session.send('Debugger.setBreakpointByUrl', {
        urlRegex: `/${location.file.replaceAll('.', '\\.')}$`,
        lineNumber: location.line, condition: location.condition,
      });
      expressions.set(breakpoint.breakpointId, location.expression);
    }
    const on = session.on.bind(session);
    session.on = (event, handler) => on(event, event !== 'Debugger.paused' ? handler : async paused => {
      const id = paused.hitBreakpoints.find(id => expressions.has(id));
      if (!id) return handler(paused);
      console.log(JSON.stringify({ observation: 'attention-cache', phase: 'capture', boundary: expressions.get(id) }));
      try {
        const result = await session.send('Debugger.evaluateOnCallFrame', {
          callFrameId: paused.callFrames[0].callFrameId, expression: expressions.get(id), returnByValue: true,
        });
        if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
      } catch (error) {
        console.error(`Attention observation failed: ${error.message}`);
        await session.send('Debugger.evaluateOnCallFrame', {
          callFrameId: paused.callFrames[0].callFrameId,
          expression: `attentionCacheObservation.errors.push(${JSON.stringify(error.message)})`,
          returnByValue: true,
        });
      } finally {
        await session.send('Debugger.resume');
        console.log(JSON.stringify({ observation: 'attention-cache', phase: 'resumed' }));
      }
    });
    return session;
  };
  const close = context.close.bind(context);
  context.close = async (...args) => {
    try {
      for (const page of context.pages()) {
        const capture = await page.evaluate(async () => {
          await Promise.all(attentionCacheObservationPending);
          return attentionCacheObservation;
        });
        captures.push(capture);
        assert.deepEqual(capture.errors, []);
        assert(capture.records.every(r => r.tensors.every(t => t.data)), 'Every observed tensor must settle');
      }
    } finally { await close(...args); }
  };
}
