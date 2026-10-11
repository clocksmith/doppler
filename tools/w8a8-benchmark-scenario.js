// Fixed-prefix operations through installed public model handles. This does not
// turn predetermined decode inputs into an autoregressive generation benchmark.
export async function runW8A8BenchmarkScenario(config, reference, observe) {
  const api = await import(config.apiModule);
  const { getDevice } = await import(config.deviceModule);
  const { configureDeviceMemoryBudget, inspectDeviceMemory } = await import(config.partitionsModule);
  const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
  const equal = (left, right) => JSON.stringify(left) === JSON.stringify(right);
  const policy = reference.suite.benchmark;
  const serial = config.residencyPolicy === 'serialized-resident';
  requireValue(serial || config.residencyPolicy === 'paired-resident', 'Benchmark residency policy must be explicit.');
  const report = { passed: false, outputs: [], pairs: [], cleanup: [], loads: [], residencyPolicy: config.residencyPolicy,
    scope: 'Fixed shared input prefixes and predetermined decode tokens, including required logits readback and public prefill snapshot cost; not complete-generation latency.' };
  const handles = new Map();
  const options = { ...reference.suite.generation, useChatTemplate: false };
  async function load(model) {
    const variant = reference.executionVariants?.find(row => row.modelIdentity === model.manifestDigest);
    const sourceIdentity = variant ? variant.independentSourceModelIdentity : model.manifestDigest;
    requireValue(reference.lanes.some(lane => lane.modelIdentity === sourceIdentity), 'Unbound benchmark model.');
    const started = performance.now();
    const handle = await api.load({ url: model.modelUrl }, { runtimeConfig: config.runtimeConfig });
    handles.set(model.id, handle);
    report.loads.push({ ...model, loadMs: performance.now() - started, resolvedSession: handle.advanced.getResolvedRuntimeSession() });
    return handle;
  }
  async function unload(model) {
    const handle = handles.get(model.id);
    await handle.unload();
    handles.delete(model.id);
  }
  async function execute(model, handle, item) {
    handle.resetGenerationState();
    requireValue(equal(handle.advanced.tokenizePrompt(item.text, options), item.inputIds), 'Benchmark input token identities differ.');
    const start = performance.now();
    const first = await handle.advanced.prefillWithLogits(item.text, options);
    first.cache.destroy();
    requireValue(!first.linearAttention || first.linearAttention.layers?.size === 0, 'Recurrent benchmark snapshot is unsupported.');
    await getDevice().queue.onSubmittedWorkDone();
    const prefillEnd = performance.now();
    let logits = first.logits;
    for (const token of item.decodeIds) logits = (await handle.advanced.decodeStepLogits([token], options)).logits;
    const end = performance.now();
    requireValue(logits instanceof Float32Array && logits.every(Number.isFinite), 'Benchmark returned nonfinite logits.');
    const binding = report.loads.findLast(row => row.id === model.id);
    if (!binding.observedExecution) {
      const stats = handle.advanced.getStats();
      binding.observedExecution = { kernelPathId: stats.kernelPathId, executionPlan: stats.executionPlan };
    }
    requireValue(first.tokens.length === item.inputIds.length, 'Actual prefill shape differs from frozen work.');
    return { id: model.id, prefillMs: prefillEnd - start, decodeMs: end - prefillEnd,
      completeOperationMs: end - start, actualInputTokens: first.tokens.length,
      actualDecodeSteps: item.decodeIds.length, finalVocabulary: logits.length };
  }
  try {
    await configureDeviceMemoryBudget({ maxBytes: reference.suite.lifecycle.maxGpuBufferBytes });
    requireValue(config.comparisonModels.length === 2, 'A paired benchmark requires exactly two explicitly bound models.');
    if (!serial) for (const model of config.comparisonModels) await load(model);
    report.memoryAfterInitialLoad = inspectDeviceMemory();
    for (const item of reference.fixedWork) {
      const firstIndex = serial ? 0 : -policy.warmupRuns;
      for (let index = firstIndex; index < policy.pairedRuns; index++) {
        const order = index % 2 === 0 ? config.comparisonModels : [...config.comparisonModels].reverse();
        const pair = { inputTokens: item.inputIds.length, decodeTokens: item.decodeIds.length, index,
          warmup: index < 0, residencyWarmups: [], runs: [] };
        report.pairs.push(pair); // Retain partial pairs if acquisition or execution fails.
        for (const model of order) {
          const handle = serial ? await load(model) : handles.get(model.id);
          if (serial) {
            for (let warmup = 0; warmup < policy.warmupRuns; warmup++) {
              pair.residencyWarmups.push(await execute(model, handle, item));
            }
          }
          pair.runs.push(await execute(model, handle, item));
          if (serial) await unload(model);
        }
        await observe({ stage: 'fixed-work-pair', inputTokens: pair.inputTokens, decodeTokens: pair.decodeTokens, index, warmup: pair.warmup });
      }
    }
    report.memory = inspectDeviceMemory();
    requireValue(report.memory.peakBytes <= reference.suite.lifecycle.maxGpuBufferBytes, 'Shared allocation ceiling exceeded.');
    report.passed = true;
  } catch (error) { report.error = { name: error.name, message: error.message, stack: error.stack }; }
  finally {
    for (const handle of handles.values()) {
      try { await handle.unload(); } catch (error) { report.cleanup.push(error.message); }
    }
    report.passed &&= report.cleanup.length === 0;
  }
  return report;
}
