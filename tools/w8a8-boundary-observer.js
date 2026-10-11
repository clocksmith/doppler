// Internal observation on installed code, separate from public model qualification.
export async function runW8A8BoundaryScenario(config, reference, observe) {
  const base = new URL(config.poolObservationModule, globalThis.location?.href ?? import.meta.url);
  const moduleUrl = path => new URL('../' + path, base).href;
  const { PipelineGenerator } = await import(moduleUrl('inference/pipelines/text/generator.js'));
  const { OperatorEventEmitter } = await import(moduleUrl('inference/pipelines/text/operator-events.js'));
  const { createDefaultCaptureConfig, validateCaptureConfig } = await import(moduleUrl('debug/index.js'));
  const api = await import(config.apiModule);
  const { getDevice } = await import(config.deviceModule);
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  const report = { passed: false, outputs: [], cleanup: [], scope: 'Internal installed decode observation; no accuracy-reference activation' };
  const document = reference.documents[0];
  assert(document.firstText === '' && document.inputIds[0] === 2 && document.inputIds[1] === 236776,
    'Boundary context differs from the selected failing teacher-forced prefix.');
  const boundaryStage = config.boundaryStage === undefined ? 'qkv' : config.boundaryStage;
  assert(['qkv', 'layer-output', 'projections'].includes(boundaryStage), 'Unknown boundary observation stage');
  const boundaryLayer = boundaryStage === 'projections' ? config.boundaryLayer : 0;
  assert(Number.isInteger(boundaryLayer) && boundaryLayer >= 0 && boundaryLayer < 18, 'Invalid selected boundary layer');
  const opIds = ['post_input_norm', 'q_proj', 'k_proj', 'v_proj'].map(stage => `layer.${boundaryLayer}.attn.${stage}`);
  if (boundaryStage === 'layer-output') {
    opIds.push(...Array.from({ length: 18 }, (_, layer) => `layer.${layer}.layer.out`));
  }
  if (boundaryStage === 'projections') {
    opIds.push(...['attn.core_out', 'attn.out', 'attn.post_attn', 'ffn.in',
      'ffn.gate', 'ffn.up', 'ffn.act', 'ffn.out', 'layer.out']
      .map(stage => `layer.${boundaryLayer}.${stage}`));
  }
  report.boundaryStage = boundaryStage;
  report.boundaryLayer = boundaryLayer;
  report.requestedOpIds = opIds;
  const prototype = PipelineGenerator.prototype;
  const original = prototype._decodeStepToLogits;
  let session;
  try {
    session = await api.load({ url: config.modelUrl }, { runtimeConfig: config.runtimeConfig });
    const options = { ...config.generation, useChatTemplate: false };
    const run = async () => {
      session.resetGenerationState();
      const first = await session.advanced.prefillWithLogits(document.firstText, options);
      try { assert(JSON.stringify(first.tokens) === '[2]', 'Initial context is not BOS alone'); }
      finally { first.cache.destroy(); }
      assert(!first.linearAttention || first.linearAttention.layers?.size === 0, 'This observer does not own recurrent state');
      return session.advanced.decodeStepLogits([236776], options);
    };
    const baseline = await run();
    const captures = [];
    let calls = 0;
    const captureConfig = { ...createDefaultCaptureConfig(), enabled: true,
      defaultLevel: 'none', targetOpIds: opIds, targetLevel: 'full' };
    validateCaptureConfig(captureConfig);
    prototype._decodeStepToLogits = async function (...args) {
      calls++;
      const previous = this._state.operatorDiagnostics;
      const emitter = new OperatorEventEmitter({ modelHash: config.manifestDigest,
        runtimeConfigHash: null, executionPlanHash: null });
      this._state.operatorDiagnostics = { enabled: true, captureConfig, emitter, tsirFixture: null };
      try { return await original.apply(this, args); }
      finally {
        for (const row of emitter.getTimeline()) {
          if (row.capture?.data) captures.push({ ...row, capture: { ...row.capture, data: Array.from(row.capture.data) } });
        }
        this._state.operatorDiagnostics = previous;
      }
    };
    let observed;
    try { observed = await run(); }
    finally { prototype._decodeStepToLogits = original; }
    assert(calls === 1, 'Observation wrapped more than the selected decode boundary');
    const before = new Uint32Array(baseline.logits.buffer, baseline.logits.byteOffset, baseline.logits.length);
    const after = new Uint32Array(observed.logits.buffer, observed.logits.byteOffset, observed.logits.length);
    let different = 0;
    for (let index = 0; index < before.length; index++) if (before[index] !== after[index]) different++;
    report.context = [2, 236776]; report.targetToken = document.inputIds[2];
    report.baselineTargetLogit = baseline.logits[report.targetToken];
    report.observedTargetLogit = observed.logits[report.targetToken];
    report.logitCount = before.length; report.differentLogitBits = different;
    report.captures = captures;
    report.outputs.push({ context: report.context, targetToken: report.targetToken, logit: report.baselineTargetLogit });
    assert(before.length === after.length && different === 0, 'Observation changed baseline logits');
    for (const opId of opIds) assert(captures.some(row => row.opId === opId), 'Missing full capture: ' + opId);
    await getDevice().queue.onSubmittedWorkDone();
    report.passed = true;
    await observe({ stage: 'boundary-captured', captures: captures.length, differentLogitBits: different });
  } catch (error) { report.error = { name: error.name, message: error.message, stack: error.stack }; }
  finally {
    prototype._decodeStepToLogits = original;
    try { await session?.unload(); } catch (error) { report.cleanup.push(error.message); }
    report.passed &&= report.cleanup.length === 0;
  }
  return report;
}
