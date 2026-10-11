// Fixed-prefix operations through installed public model handles. This does not
// turn predetermined decode inputs into an autoregressive generation benchmark.
export async function runW8A8BenchmarkScenario(config, reference, observe) {
  const api = await import(config.apiModule);
  const { getDevice } = await import(config.deviceModule);
  const { configureDeviceMemoryBudget, inspectDeviceMemory } = await import(config.partitionsModule);
  const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
  const equal = (left, right) => JSON.stringify(left) === JSON.stringify(right);
  const report = { passed: false, outputs: [], pairs: [], cleanup: [], loads: [],
    scope: 'Fixed shared input prefixes and predetermined decode tokens, including required logits readback and public prefill snapshot cost; not complete-generation latency.' };
  const handles = new Map();
  const options = { ...reference.suite.generation, useChatTemplate: false };
  try {
    await configureDeviceMemoryBudget({ maxBytes: reference.suite.lifecycle.maxGpuBufferBytes });
    requireValue(config.comparisonModels.length === 2, 'A paired benchmark requires exactly two explicitly bound models.');
    for (const model of config.comparisonModels) {
      requireValue(reference.lanes.some(lane => lane.modelIdentity === model.manifestDigest), 'Unbound benchmark model.');
      const started = performance.now();
      const handle = await api.load({ url: model.modelUrl }, { runtimeConfig: config.runtimeConfig });
      handles.set(model.id, handle);
      report.loads.push({ ...model, loadMs: performance.now() - started, resolvedSession: handle.advanced.getResolvedRuntimeSession() });
    }
    report.memoryAfterLoad = inspectDeviceMemory();
    const policy = reference.suite.benchmark;
    for (const item of reference.fixedWork) {
      for (let index = -policy.warmupRuns; index < policy.pairedRuns; index++) {
        const order = index % 2 === 0 ? config.comparisonModels : [...config.comparisonModels].reverse();
        const pair = { inputTokens: item.inputIds.length, decodeTokens: item.decodeIds.length, index, warmup: index < 0, runs: [] };
        for (const model of order) {
          const handle = handles.get(model.id);
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
          if (index === -policy.warmupRuns) {
            const stats = handle.advanced.getStats();
            report.loads.find(row => row.id === model.id).observedExecution = { kernelPathId: stats.kernelPathId, executionPlan: stats.executionPlan };
          }
          pair.runs.push({ id: model.id, prefillMs: prefillEnd - start, decodeMs: end - prefillEnd,
            completeOperationMs: end - start, actualInputTokens: first.tokens.length,
            actualDecodeSteps: item.decodeIds.length, finalVocabulary: logits.length });
          requireValue(first.tokens.length === item.inputIds.length, 'Actual prefill shape differs from frozen work.');
        }
        report.pairs.push(pair);
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
