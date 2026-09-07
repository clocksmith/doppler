// The same scenario runs against installed Node/Bun and browser module URLs.
// All tensor execution stays in the installed runtime; comparisons are observations.
export function compareGenerationTokens(row, actual, promptTokens) {
  const expected = row.generatedTokenIds, tokens = actual.tokenIds;
  if (![expected, tokens, promptTokens, row.promptTokenIds].every(values => Array.isArray(values)
    && values.every(token => Number.isSafeInteger(token) && token >= 0))) {
    throw new Error('Generation reference comparison requires complete integer token arrays.');
  }
  const mismatch = Array.from({ length: Math.max(expected.length, tokens.length) }, (_, index) => index)
    .find(index => expected[index] !== tokens[index]);
  const promptEqual = JSON.stringify(promptTokens) === JSON.stringify(row.promptTokenIds);
  return { queryIndex: row.queryIndex, passed: mismatch === undefined && promptEqual,
    firstTokenMismatch: mismatch ?? null, promptEqual, expectedTokens: expected.length, observedTokens: tokens.length };
}

export async function runGenerationQualificationScenario(config, reference, observe) {
  const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
  const api = await import(config.apiModule);
  const { getDevice, destroyDevice } = await import(config.deviceModule);
  const { observeInitialExecutionIdentity } = await import(config.identityModule);
  const report = { passed: false, outputs: [], stage: 'load', cleanup: [] };
  let session;
  async function open() {
    const started = performance.now();
    session = await api.load({ url: config.modelUrl }, { runtimeConfig: config.runtimeConfig });
    const source = session.manifest.artifactIdentity;
    requireValue(source.sourceRepo === reference.source.model && source.sourceRevision === reference.source.revision,
      'Generation source identity differs from the frozen reference.');
    const identity = observeInitialExecutionIdentity(session.advanced.getResolvedRuntimeSession());
    return { loadMs: performance.now() - started, identity, manifest: session.manifest };
  }
  async function generate(row) {
    session.resetGenerationState();
    const promptTokens = session.advanced.tokenizePrompt(row.prompt, { useChatTemplate: config.generation.useChatTemplate });
    const started = performance.now();
    const result = await session.generateWithEvidence(row.prompt, config.generation);
    return { queryIndex: row.queryIndex, elapsedMs: performance.now() - started,
      result, comparison: compareGenerationTokens(row, result, promptTokens) };
  }
  try {
    report.open = await open(); await observe({ stage: 'loaded', loadMs: report.open.loadMs });
    report.stage = 'source-reference';
    for (let repeat = 0; repeat < config.repeatRuns; repeat++) {
      for (const row of reference.outputs) {
        const output = { repeat, ...await generate(row) };
        report.outputs.push(output); await observe({ stage: 'generated', ...output.comparison, elapsedMs: output.elapsedMs });
        requireValue(output.comparison.passed, 'Frozen generation token reference failed.');
      }
    }
    const first = reference.outputs[0];
    report.stage = 'cancellation';
    session.resetGenerationState();
    const controller = new AbortController();
    const cancelStarted = performance.now(); let chunks = 0;
    for await (const _chunk of session.generate(first.prompt, { ...config.generation, signal: controller.signal })) {
      chunks++;
      if (chunks === config.cancellation.afterChunks) controller.abort();
    }
    const stats = session.advanced.getStats();
    report.cancellation = { chunks, elapsedMs: performance.now() - cancelStarted, stats };
    requireValue(controller.signal.aborted && stats.stopReason === 'aborted', 'Generation did not observe cancellation.');
    requireValue(report.cancellation.elapsedMs <= config.cancellation.maxElapsedMs, 'Generation cancellation exceeded the explicit budget.');
    report.afterCancellation = await generate(first);
    requireValue(report.afterCancellation.comparison.passed, 'Generation failed after cancellation.');
    await observe({ stage: 'cancellation-recovered' });
    report.stage = 'device-loss';
    const device = getDevice(); device.destroy(); await device.lost;
    try { await session.generateWithEvidence(first.prompt, config.generation); }
    catch (error) { report.lostDeviceRejection = { name: error.name, message: error.message }; }
    requireValue(report.lostDeviceRejection, 'Generation on the destroyed device must fail.');
    await session.unload(); session = null; destroyDevice();
    report.reopen = await open();
    requireValue(getDevice() !== device, 'Recovery reused the destroyed device.');
    report.afterDeviceLoss = await generate(first);
    requireValue(report.afterDeviceLoss.comparison.passed, 'Generation failed after device recovery.');
    await observe({ stage: 'device-loss-recovered' });
    report.passed = true; report.stage = 'complete';
  } catch (error) { report.error = { name: error.name, message: error.message, stack: error.stack }; }
  finally {
    for (const close of [() => session?.unload(), () => destroyDevice()]) {
      try { await close(); } catch (error) { report.cleanup.push(error.message); }
    }
    report.passed &&= report.cleanup.length === 0;
  }
  return report;
}
