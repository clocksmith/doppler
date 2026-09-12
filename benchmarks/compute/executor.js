import { requireCondition } from './contract.js';

const TIMESTAMP_BYTES = 8;
const NANOSECONDS_PER_MILLISECOND = 1000000;

// Only declared buffers, bindings, dispatches, and submission grouping enter
// this executor. Mathematical operations belong to the workload adapter.
export async function preparePlan(device, inputPlan, data, sources, shared) {
  const plan = structuredClone(inputPlan);
  const owned = [];
  const resources = new Map();
  const pipelineRecords = [];
  const started = performance.now();
  let closed = false;
  function own(value) { owned.push(value); return value; }
  function close() {
    if (closed) return;
    closed = true;
    for (const object of owned.reverse()) {
      if (object.mapState === 'mapped') object.unmap();
      object.destroy();
    }
  }
  try {
    requireCondition(['batch', 'dispatch'].includes(plan.submission), 'Unknown submission policy.');
    requireCondition(plan.steps.length > 0 && plan.outputBytes === (plan.outputElements + shared.guardElements) * 4, 'Invalid output envelope.');
    let allocatedBytes = 0;
    for (const [id, resource] of Object.entries(plan.resources)) {
      requireCondition(Number.isInteger(resource.bytes) && resource.bytes > 0 && resource.bytes % 4 === 0, `Invalid resource size: ${id}`);
      requireCondition(['storage', 'uniform'].includes(resource.kind), `Unknown resource kind: ${id}`);
      const limit = resource.kind === 'uniform' ? device.limits.maxUniformBufferBindingSize : device.limits.maxStorageBufferBindingSize;
      requireCondition(resource.bytes <= limit && resource.bytes <= device.limits.maxBufferSize, `Resource exceeds device limit: ${id}`);
      const role = resource.kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE;
      resources.set(id, own(device.createBuffer({ label: `${plan.id}/${id}`, size: resource.bytes, usage: role | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })));
      allocatedBytes += resource.bytes;
    }
    requireCondition(resources.has(plan.output) && plan.resources[plan.output].bytes === plan.outputBytes, 'Output resource does not match the declared size.');
    const readback = own(device.createBuffer({ label: `${plan.id}/readback`, size: plan.outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    allocatedBytes += plan.outputBytes;
    const allocationMs = performance.now() - started;
    let uploadedBytes = 0;
    let writeBufferCalls = 0;
    const uploadStart = performance.now();
    for (const [id, bytes] of Object.entries(data)) {
      if (!resources.has(id)) continue;
      requireCondition(bytes.byteLength === plan.resources[id].bytes, `Input byte length differs from resource: ${id}`);
      device.queue.writeBuffer(resources.get(id), 0, bytes);
      uploadedBytes += bytes.byteLength;
      writeBufferCalls += 1;
    }
    await device.queue.onSubmittedWorkDone();
    const uploadMs = performance.now() - uploadStart;
    const preparedSteps = [];
    for (const step of plan.steps) {
      requireCondition(step.dispatch.length === 3 && step.dispatch.every(n => Number.isSafeInteger(n) && n > 0 && n <= device.limits.maxComputeWorkgroupsPerDimension), `Invalid dispatch: ${step.id}`);
      requireCondition(typeof sources[step.source] === 'string', `Missing shader source: ${step.source}`);
      const entries = step.bindings.map((binding, index) => {
        requireCondition(resources.has(binding.resource), `Unknown binding resource: ${binding.resource}`);
        return { binding: index, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding.type } };
      });
      const start = performance.now();
      const module = device.createShaderModule({ label: step.source, code: sources[step.source] });
      const compilationInfo = await module.getCompilationInfo();
      const messages = Array.from(compilationInfo.messages, message => ({ type: message.type, message: message.message, lineNum: message.lineNum, linePos: message.linePos }));
      requireCondition(!messages.some(message => message.type === 'error'), `Shader compilation failed: ${step.source}: ${JSON.stringify(messages)}`);
      const layout = device.createBindGroupLayout({ entries });
      const pipeline = await device.createComputePipelineAsync({ label: `${plan.id}/${step.id}`, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: step.entryPoint, constants: step.constants } });
      const pipelineMs = performance.now() - start;
      const bindStart = performance.now();
      const bindGroup = device.createBindGroup({ layout, entries: step.bindings.map((binding, index) => ({ binding: index, resource: { buffer: resources.get(binding.resource) } })) });
      const bindMs = performance.now() - bindStart;
      pipelineRecords.push({ stepId: step.id, pipelineMs, bindMs, messages });
      preparedSteps.push({ ...step, pipeline, bindGroup });
    }
    const poison = new Float32Array(plan.outputElements + shared.guardElements);
    poison.fill(NaN, 0, plan.outputElements);
    poison.fill(shared.guardValue, plan.outputElements);
    for (const id of plan.resetResources) requireCondition(resources.has(id) && plan.resources[id].bytes === poison.byteLength, 'Reset resource has a different output envelope.');
    return {
      plan,
      preparation: { allocationMs, uploadMs, uploadedBytes, writeBufferCalls, allocatedBytes, pipelineRecords, totalMs: performance.now() - started, cacheState: 'fresh-JS-pipelines; browser-and-driver-cache-uncontrolled' },
      close,
      async run({ repetitions, profile = false, trace = false }) {
        requireCondition(!closed, 'Cannot execute a closed plan.');
        requireCondition(Number.isSafeInteger(repetitions) && repetitions > 0, 'Invalid repetition count.');
        requireCondition(!profile || device.features.has('timestamp-query'), 'GPU timestamp profiling is unavailable.');
        let querySet;
        let queryResolve;
        let queryReadback;
        const queryCount = repetitions * preparedSteps.length * 2;
        try {
          if (profile) {
            querySet = device.createQuerySet({ type: 'timestamp', count: queryCount });
            queryResolve = device.createBuffer({ size: queryCount * TIMESTAMP_BYTES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            queryReadback = device.createBuffer({ size: queryCount * TIMESTAMP_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          }
          const resetStart = performance.now();
          for (const id of plan.resetResources) device.queue.writeBuffer(resources.get(id), 0, poison);
          await device.queue.onSubmittedWorkDone();
          const resetMs = performance.now() - resetStart;
          const counters = { dispatches: 0, submissions: 0, workgroups: 0, readbackSubmissions: 0, outputBytesRead: 0, resetWriteBufferCalls: plan.resetResources.length, resetBytesUploaded: poison.byteLength * plan.resetResources.length };
          const traceRows = [];
          let encodeMs = 0;
          let submitCpuMs = 0;
          let encoder = null;
          let queryIndex = 0;
          function submit() {
            const finishStart = performance.now();
            const commands = encoder.finish();
            encodeMs += performance.now() - finishStart;
            const submitStart = performance.now();
            device.queue.submit([commands]);
            submitCpuMs += performance.now() - submitStart;
            counters.submissions += 1;
            encoder = null;
          }
          const runStart = performance.now();
          for (let repetition = 0; repetition < repetitions; repetition += 1) {
            for (const step of preparedSteps) {
              const encodeStart = performance.now();
              encoder ??= device.createCommandEncoder({ label: plan.id });
              const descriptor = profile ? { timestampWrites: { querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } } : {};
              const pass = encoder.beginComputePass(descriptor);
              pass.setPipeline(step.pipeline);
              pass.setBindGroup(0, step.bindGroup);
              pass.dispatchWorkgroups(...step.dispatch);
              pass.end();
              encodeMs += performance.now() - encodeStart;
              counters.dispatches += 1;
              counters.workgroups += step.dispatch.reduce((a, b) => a * b, 1);
              if (trace || profile) traceRows.push({ repetition, stepId: step.id, source: step.source, dispatch: step.dispatch, submission: counters.submissions, queryIndex: profile ? queryIndex : null });
              queryIndex += 2;
              if (plan.submission === 'dispatch') submit();
            }
          }
          if (encoder) submit();
          const waitStart = performance.now();
          await device.queue.onSubmittedWorkDone();
          const completed = performance.now();
          const readStart = performance.now();
          const copy = device.createCommandEncoder();
          copy.copyBufferToBuffer(resources.get(plan.output), 0, readback, 0, plan.outputBytes);
          if (profile) {
            copy.resolveQuerySet(querySet, 0, queryCount, queryResolve, 0);
            copy.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, queryCount * TIMESTAMP_BYTES);
          }
          device.queue.submit([copy.finish()]);
          counters.readbackSubmissions += 1;
          counters.outputBytesRead = plan.outputBytes;
          await readback.mapAsync(GPUMapMode.READ);
          let output;
          try { output = new Float32Array(readback.getMappedRange().slice(0)); } finally { readback.unmap(); }
          const readbackMs = performance.now() - readStart;
          let gpu = null;
          if (profile) {
            await queryReadback.mapAsync(GPUMapMode.READ);
            try {
              const stamps = new BigUint64Array(queryReadback.getMappedRange());
              const passTimes = traceRows.map(row => {
                const begin = stamps[row.queryIndex];
                const end = stamps[row.queryIndex + 1];
                requireCondition(end >= begin, 'GPU timestamp order is invalid.');
                return { ...row, beginNs: String(begin), endNs: String(end), gpuMs: Number(end - begin) / NANOSECONDS_PER_MILLISECOND };
              });
              gpu = { source: 'timestamp-query', instrumentation: 'separate-diagnostic-pass', unit: 'ms', summedPassMs: passTimes.reduce((sum, row) => sum + row.gpuMs, 0), passTimes, zeroDurationPasses: passTimes.filter(row => row.gpuMs === 0).length };
            } finally { queryReadback.unmap(); }
          }
          return { output, wallMs: completed - runStart, encodeMs, submitCpuMs, waitAfterLastSubmitMs: completed - waitStart, resetMs, readbackMs, counters, gpu, trace: trace ? traceRows : null };
        } finally {
          if (readback.mapState === 'mapped') readback.unmap();
          if (queryReadback?.mapState === 'mapped') queryReadback.unmap();
          queryReadback?.destroy();
          queryResolve?.destroy();
          querySet?.destroy();
        }
      },
    };
  } catch (error) { close(); throw error; }
}
