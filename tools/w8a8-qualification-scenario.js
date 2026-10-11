// Installed public model operations; numerical comparisons are observations only.
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };

export function summarizeLogits(logits, target) {
  requireValue(logits instanceof Float32Array && logits.length > target, 'Missing full vocabulary logits.');
  let maximum = -Infinity, top1 = -1;
  for (let index = 0; index < logits.length; index++) {
    requireValue(Number.isFinite(logits[index]), 'Nonfinite logits.');
    if (logits[index] > maximum) { maximum = logits[index]; top1 = index; }
  }
  let sum = 0;
  for (const value of logits) sum += Math.exp(value - maximum);
  return { targetLogit: logits[target], nll: maximum + Math.log(sum) - logits[target], top1, vocabulary: logits.length };
}

export async function qualifySubmittedGeneration({ queue, run, stats, resources }) {
  const before = structuredClone(resources());
  const controller = new AbortController();
  const reason = new Error('Generation qualification cancellation after GPU submission');
  const originalSubmit = queue.submit;
  let submissions = 0, rejection = null;
  queue.submit = function (...args) {
    const result = originalSubmit.apply(this, args);
    submissions++;
    if (!controller.signal.aborted) controller.abort(reason);
    return result;
  };
  try { await run(controller.signal); }
  catch (error) { rejection = error; }
  finally {
    queue.submit = originalSubmit;
    await queue.onSubmittedWorkDone();
  }
  const after = structuredClone(resources());
  const aborted = stats().stopReason === 'aborted';
  const abortRejection = rejection === reason || rejection?.name === 'AbortError'
    || String(rejection?.message).includes(reason.message);
  const evidence = { submissions, before, after, stopReason: stats().stopReason,
    rejection: rejection ? { name: rejection.name, message: rejection.message } : null };
  try {
    requireValue(submissions > 0 && controller.signal.aborted, 'Cancellation must follow an actual GPU submission.');
    requireValue(rejection ? abortRejection : aborted, 'Submitted generation did not report cancellation.');
    requireValue(after.active.bytes <= before.active.bytes && after.active.count <= before.active.count
      && after.deferredCleanup.bytes === 0 && after.deferredCleanup.count === 0,
    'Submitted cancellation retained active or deferred resources.');
  } catch (error) { error.evidence = evidence; throw error; }
  return { passed: true, ...evidence };
}

export async function runW8A8QualificationScenario(config, reference, observe) {
  const api = await import(config.apiModule);
  const { getDevice } = await import(config.deviceModule);
  const { configureDeviceMemoryBudget, inspectDeviceMemory } = await import(config.partitionsModule);
  const { getBufferPool } = await import(config.poolObservationModule);
  const report = { passed: false, outputs: [], teacherForced: [], fixedWork: [], cleanup: [], stage: 'load' };
  let session;
  const generation = reference.suite.generation;
  const variant = reference.executionVariants?.find(row => row.modelIdentity === config.manifestDigest);
  const sourceIdentity = variant ? variant.independentSourceModelIdentity : config.manifestDigest;
  const lane = reference.lanes.find(row => row.modelIdentity === sourceIdentity);
  report.referenceBinding = { modelIdentity: config.manifestDigest, independentSourceModelIdentity: sourceIdentity, variant: variant ?? null };
  requireValue(lane, 'No independently bound model reference.');
  async function generate(task, referenceRow) {
    session.resetGenerationState();
    const inputIds = session.advanced.tokenizePrompt(task.messages, generation);
    requireValue(same(inputIds, referenceRow.inputIds), 'Source/application prompt tokens differ.');
    const start = performance.now();
    const result = await session.generateWithEvidence(task.messages, generation);
    const elapsedMs = performance.now() - start;
    const stats = session.advanced.getStats();
    const naturalStop = ['eos-token', 'stop-token'].includes(stats.stopReason);
    return { id: task.id, inputIds, elapsedMs, result, stats, naturalStop,
      taskPassed: naturalStop && result.outputText.replace(/^[ \t\r\n\f\v]+|[ \t\r\n\f\v]+$/g, '') === task.expected,
      sourceTokensEqual: same(result.tokenIds, referenceRow.tokenIds) };
  }
  try {
    await configureDeviceMemoryBudget({ maxBytes: reference.suite.lifecycle.maxGpuBufferBytes });
    const started = performance.now();
    session = await api.load({ url: config.modelUrl }, { runtimeConfig: config.runtimeConfig });
    report.open = { loadMs: performance.now() - started, resolvedSession: session.advanced.getResolvedRuntimeSession(), manifest: session.manifest };
    await observe({ stage: 'loaded', loadMs: report.open.loadMs });
    for (const [index, task] of reference.suite.quality.tasks.entries()) {
      const row = await generate(task, lane.tasks[index]);
      report.outputs.push(row);
      await observe({ stage: 'quality', id: task.id, passed: row.taskPassed, sourceTokensEqual: row.sourceTokensEqual });
    }
    // Decode fixed teacher-forced tokens; no generated token feeds the control.
    for (const document of reference.documents) {
      session.resetGenerationState();
      requireValue(same(session.advanced.tokenizePrompt(document.firstText, { useChatTemplate: false }), document.inputIds.slice(0, 1)), 'Teacher-forced initial token differs.');
      const first = await session.advanced.prefillWithLogits(document.firstText, { ...generation, useChatTemplate: false });
      first.cache.destroy();
      requireValue(!first.linearAttention || first.linearAttention.layers?.size === 0, 'This qualification does not own recurrent snapshot disposal.');
      const rows = [summarizeLogits(first.logits, document.inputIds[1])];
      for (let index = 1; index + 1 < document.inputIds.length; index++) {
        const step = await session.advanced.decodeStepLogits([document.inputIds[index]], generation);
        rows.push(summarizeLogits(step.logits, document.inputIds[index + 1]));
      }
      report.teacherForced.push({ document: document.document, inputIds: document.inputIds, rows });
      await observe({ stage: 'teacher-forced', document: document.document, targets: rows.length });
    }
    report.repeated = [];
    for (let repeat = 0; repeat < reference.suite.lifecycle.repeatedRequests; repeat++) {
      report.repeated.push(await generate(reference.suite.quality.tasks[0], lane.tasks[0]));
    }
    session.resetGenerationState();
    const controller = new AbortController(); let callbacks = 0;
    for await (const _chunk of session.generate(reference.suite.quality.tasks[0].messages[0].content,
      { ...generation, signal: controller.signal, onToken() { callbacks++; controller.abort(); } })) { /* owned stream drained */ }
    report.cancellation = { callbacks, aborted: controller.signal.aborted, stats: session.advanced.getStats() };
    requireValue(callbacks > 0 && controller.signal.aborted && report.cancellation.stats.stopReason === 'aborted', 'First-token callback cancellation was not observed.');
    report.afterCancellation = await generate(reference.suite.quality.tasks[0], lane.tasks[0]);
    requireValue(same(report.afterCancellation.result.tokenIds, report.outputs[0].result.tokenIds), 'Resident recovery changed deterministic generation.');
    requireValue(report.repeated.every(row => same(row.result.tokenIds, report.outputs[0].result.tokenIds)), 'Resident reuse changed deterministic generation.');
    session.resetGenerationState();
    await getDevice().queue.onSubmittedWorkDone();
    report.submittedCancellation = await qualifySubmittedGeneration({
      queue: getDevice().queue,
      run: async signal => {
        for await (const _chunk of session.generate(reference.suite.quality.tasks[0].messages[0].content,
          { ...generation, signal })) { /* runtime owns the cancelled stream */ }
      },
      stats: () => session.advanced.getStats(),
      resources: () => getBufferPool().getStats().resources,
    });
    report.afterSubmittedCancellation = await generate(reference.suite.quality.tasks[0], lane.tasks[0]);
    requireValue(same(report.afterSubmittedCancellation.result.tokenIds, report.outputs[0].result.tokenIds),
      'Submitted cancellation changed subsequent deterministic generation.');
    await getDevice().queue.onSubmittedWorkDone();
    report.memory = { device: inspectDeviceMemory(), pool: getBufferPool().getStats() };
    requireValue(report.memory.device.peakBytes <= reference.suite.lifecycle.maxGpuBufferBytes, 'Allocation ceiling exceeded.');
    report.summary = { correctTasks: report.outputs.filter(row => row.taskPassed).length,
      taskCount: report.outputs.length, sourceTokenMatches: report.outputs.filter(row => row.sourceTokensEqual).length,
      meanNll: report.teacherForced.flatMap(row => row.rows).reduce((sum, row) => sum + row.nll, 0)
        / report.teacherForced.reduce((sum, row) => sum + row.rows.length, 0),
      qualityAccepted: false, lifecyclePassed: true };
    report.passed = true; report.stage = 'execution-complete';
  } catch (error) { report.error = { name: error.name, message: error.message, stack: error.stack, evidence: error.evidence ?? null }; }
  finally {
    try {
      await session?.unload();
      if (getDevice()) {
        await getDevice().queue.onSubmittedWorkDone();
        report.afterUnload = { device: inspectDeviceMemory(), pool: getBufferPool().getStats() };
        requireValue(report.afterUnload.pool.resources.active.bytes === 0 && report.afterUnload.pool.resources.deferredCleanup.bytes === 0, 'Owned temporary buffers did not settle.');
      }
    } catch (error) { report.cleanup.push(error.message); }
    report.passed &&= report.cleanup.length === 0;
    if (report.summary) report.summary.lifecyclePassed &&= report.cleanup.length === 0;
  }
  return report;
}
