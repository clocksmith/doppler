import assert from 'node:assert/strict';

// ============================================================================
// 1. Stage Names
// ============================================================================
{
  const {
    STAGES,
    OPERATOR_CLASSES,
    PROBE_TO_CANONICAL,
    getOperatorClass,
    canonicalizeProbeStage,
    isValidStage,
  } = await import('../../src/inference/pipelines/text/stage-names.js');

  assert.equal(typeof STAGES, 'object', 'STAGES must be an object');
  assert.ok(Object.isFrozen(STAGES), 'STAGES must be frozen');
  assert.ok(Object.keys(STAGES).length >= 40, 'STAGES must have at least 40 entries');

  assert.equal(STAGES.EMBED_OUT, 'embed.out');
  assert.equal(STAGES.ATTN_Q_PROJ, 'attn.q_proj');
  assert.equal(STAGES.FFN_OUT, 'ffn.out');
  assert.equal(STAGES.LOGITS_OUT, 'logits.out');
  assert.equal(STAGES.FINAL_NORM_OUT, 'final_norm.out');

  assert.ok(Object.isFrozen(OPERATOR_CLASSES), 'OPERATOR_CLASSES must be frozen');
  assert.equal(OPERATOR_CLASSES.EMBEDDING, 'embedding');
  assert.equal(OPERATOR_CLASSES.ATTENTION, 'attention');
  assert.equal(OPERATOR_CLASSES.LOGITS, 'logits');

  assert.ok(Object.isFrozen(PROBE_TO_CANONICAL), 'PROBE_TO_CANONICAL must be frozen');
  assert.equal(PROBE_TO_CANONICAL.q_proj, 'attn.q_proj');
  assert.equal(PROBE_TO_CANONICAL.embed_out, 'embed.out');
  assert.equal(PROBE_TO_CANONICAL.logits, 'logits.out');
  assert.equal(PROBE_TO_CANONICAL.ffn_gate, 'ffn.gate');
  assert.equal(PROBE_TO_CANONICAL.layer_out, 'layer.out');
  assert.equal(PROBE_TO_CANONICAL.q_norm, 'attn.q_norm');
  assert.equal(PROBE_TO_CANONICAL.k_norm, 'attn.k_norm');

  assert.equal(getOperatorClass('attn.q_proj'), 'projection');
  assert.equal(getOperatorClass('attn.softmax'), 'attention');
  assert.equal(getOperatorClass('ffn.normed'), 'normalization');
  assert.equal(getOperatorClass('logits.out'), 'logits');
  assert.equal(getOperatorClass('embed.out'), 'embedding');
  assert.equal(getOperatorClass('nonexistent'), null);

  assert.equal(canonicalizeProbeStage('q_proj'), 'attn.q_proj');
  assert.equal(canonicalizeProbeStage('q_norm'), 'attn.q_norm');
  assert.equal(canonicalizeProbeStage('k_norm'), 'attn.k_norm');
  assert.equal(canonicalizeProbeStage('logits'), 'logits.out');
  assert.equal(canonicalizeProbeStage('nonexistent'), null);

  assert.equal(isValidStage('attn.q_proj'), true);
  assert.equal(isValidStage('logits.out'), true);
  assert.equal(isValidStage('not_a_stage'), false);

  console.log('  stage-names: ok');
}

// ============================================================================
// 2. Operator Identity
// ============================================================================
{
  const {
    buildOpId,
    buildOpIdFromProbeStage,
    buildOpIdFromExecutionStep,
    buildOperatorMeta,
    OperatorSequence,
  } = await import('../../src/inference/pipelines/text/operator-identity.js');

  assert.equal(buildOpId('attn.q_proj', 12), 'layer.12.attn.q_proj');
  assert.equal(buildOpId('attn.softmax', 0), 'layer.0.attn.softmax');
  assert.equal(buildOpId('embed.out', null), 'embed.out');
  assert.equal(buildOpId('logits.out', undefined), 'logits.out');
  assert.equal(buildOpId('embed.out'), 'embed.out');

  assert.equal(buildOpIdFromProbeStage('q_proj', 5), 'layer.5.attn.q_proj');
  assert.equal(buildOpIdFromProbeStage('q_norm', 3), 'layer.3.attn.q_norm');
  assert.equal(buildOpIdFromProbeStage('k_norm', 3), 'layer.3.attn.k_norm');
  assert.equal(buildOpIdFromProbeStage('logits'), 'logits.out');
  assert.equal(buildOpIdFromProbeStage('embed_out', null), 'embed.out');
  assert.throws(() => buildOpIdFromProbeStage('nonexistent', 0), /Unknown probe stage/);

  assert.equal(
    buildOpIdFromExecutionStep({ section: 'layer', op: 'attention', layers: [7] }),
    'layer.7.attn.out'
  );
  assert.equal(
    buildOpIdFromExecutionStep({ section: 'final', op: 'logits', layers: [] }),
    'logits.out'
  );
  assert.equal(
    buildOpIdFromExecutionStep({ section: 'layer', op: 'rmsnorm', layers: [3] }),
    'layer.3.attn.post_input_norm'
  );
  assert.equal(
    buildOpIdFromExecutionStep({ section: 'layer', op: 'ffn', layers: [1] }),
    'layer.1.ffn.out'
  );

  const meta = buildOperatorMeta('attn.q_proj', {
    layerIdx: 5,
    phase: 'decode',
    dtype: 'f16',
    tokenIndex: 42,
    shapeSignature: '1x4096',
  });
  assert.equal(meta.opId, 'layer.5.attn.q_proj');
  assert.equal(meta.stageName, 'attn.q_proj');
  assert.equal(meta.operatorClass, 'projection');
  assert.equal(meta.phase, 'decode');
  assert.equal(meta.layerIdx, 5);
  assert.equal(meta.dtype, 'f16');
  assert.equal(meta.tokenIndex, 42);
  assert.equal(meta.shapeSignature, '1x4096');
  assert.equal(meta.quantizationMode, null);

  const seq = new OperatorSequence();
  assert.equal(seq.length, 0);

  const m1 = buildOperatorMeta('embed.out', { phase: 'prefill' });
  const m2 = buildOperatorMeta('attn.q_proj', { layerIdx: 0, phase: 'prefill' });
  const m3 = buildOperatorMeta('ffn.out', { layerIdx: 0, phase: 'prefill' });

  const s1 = seq.record(m1);
  const s2 = seq.record(m2);
  const s3 = seq.record(m3);

  assert.equal(seq.length, 3);
  assert.equal(s1.sequenceIndex, 0);
  assert.equal(s2.sequenceIndex, 1);
  assert.equal(s3.sequenceIndex, 2);

  assert.deepEqual(seq.getOpById('embed.out')?.opId, 'embed.out');
  assert.equal(seq.getOpById('nonexistent'), null);
  assert.equal(seq.getOpsByLayer(0).length, 2);
  assert.equal(seq.getOpsByClass('projection').length, 1);

  seq.clear();
  assert.equal(seq.length, 0);

  console.log('  operator-identity: ok');
}

// ============================================================================
// 3. Operator Events
// ============================================================================
{
  const {
    createOperatorExecutionRecord,
    OperatorEventEmitter,
    findFirstDivergence,
  } = await import('../../src/inference/pipelines/text/operator-events.js');

  const record = createOperatorExecutionRecord({
    stageName: 'attn.q_proj',
    layerIdx: 3,
    phase: 'decode',
    dtype: 'f16',
    modelHash: 'abc',
  });
  assert.equal(record.opId, 'layer.3.attn.q_proj');
  assert.equal(record.opType, 'projection');
  assert.equal(record.stageName, 'attn.q_proj');
  assert.equal(record.phase, 'decode');
  assert.equal(record.modelHash, 'abc');
  assert.equal(record.capturePolicy, 'none');
  assert.equal(record.driftPolicyId, 'projection');
  assert.equal(record.backend, null);
  assert.deepEqual(record.captureArtifactIds, []);

  const emitter = new OperatorEventEmitter({
    modelHash: 'model1',
    enabled: true,
  });
  assert.equal(emitter.enabled, true);
  assert.equal(emitter.length, 0);

  const opId = emitter.beginOp('attn.q_proj', { layerIdx: 0, phase: 'prefill' });
  assert.equal(opId, 'layer.0.attn.q_proj');
  const endRecord = emitter.endOp({ backend: 'metal', gpuMs: 0.5 });
  assert.equal(endRecord.opId, 'layer.0.attn.q_proj');
  assert.equal(endRecord.backend, 'metal');
  assert.ok(endRecord.timing.wallMs >= 0);
  assert.equal(endRecord.timing.gpuMs, 0.5);

  emitter.emitRecord('ffn.out', { layerIdx: 0, phase: 'prefill' });
  assert.equal(emitter.length, 2);

  assert.equal(emitter.getRecordsByLayer(0).length, 2);
  assert.equal(emitter.getRecordsByPhase('prefill').length, 2);
  assert.equal(emitter.getRecordsByOpType('projection').length, 1);
  assert.equal(emitter.getRecordByOpId('layer.0.attn.q_proj')?.backend, 'metal');
  assert.equal(emitter.getRecordByOpId('nonexistent'), null);

  const json = emitter.toJSON();
  assert.equal(json.modelHash, 'model1');
  assert.equal(json.recordCount, 2);

  emitter.disable();
  assert.equal(emitter.beginOp('attn.k_proj', { layerIdx: 1 }), null);
  assert.equal(emitter.length, 2);
  emitter.enable();

  emitter.clear();
  assert.equal(emitter.length, 0);

  const baseline = [
    { opId: 'embed.out', opType: 'embedding', dtype: 'f32', captureArtifactIds: [] },
    { opId: 'layer.0.attn.q_proj', opType: 'projection', dtype: 'f16', captureArtifactIds: [] },
    { opId: 'layer.0.ffn.out', opType: 'ffn', dtype: 'f16', captureArtifactIds: [] },
  ];
  const observed = [
    { opId: 'embed.out', opType: 'embedding', dtype: 'f32', captureArtifactIds: [] },
    { opId: 'layer.0.attn.k_proj', opType: 'projection', dtype: 'f16', captureArtifactIds: [] },
    { opId: 'layer.0.ffn.out', opType: 'ffn', dtype: 'f16', captureArtifactIds: [] },
  ];
  const div = findFirstDivergence(baseline, observed);
  assert.equal(div.type, 'sequence_mismatch');
  assert.equal(div.index, 1);

  const matching = [
    { opId: 'embed.out', opType: 'embedding', dtype: 'f32', captureArtifactIds: [] },
  ];
  assert.equal(findFirstDivergence(matching, matching), null);

  const shorter = [
    { opId: 'embed.out', opType: 'embedding', dtype: 'f32', captureArtifactIds: [] },
  ];
  const longer = [
    { opId: 'embed.out', opType: 'embedding', dtype: 'f32', captureArtifactIds: [] },
    { opId: 'layer.0.attn.q_proj', opType: 'projection', dtype: 'f16', captureArtifactIds: [] },
  ];
  const lenDiv = findFirstDivergence(shorter, longer);
  assert.equal(lenDiv.type, 'length_mismatch');
  assert.equal(lenDiv.baselineLength, 1);
  assert.equal(lenDiv.observedLength, 2);

  console.log('  operator-events: ok');
}

// ============================================================================
// 4. Drift Policy
// ============================================================================
{
  const {
    getDriftTolerance,
    getDriftPolicyId,
    getOperatorClasses,
    checkDrift,
    checkPropagationBound,
  } = await import('../../src/inference/pipelines/text/drift-policy.js');

  const classes = getOperatorClasses();
  assert.ok(classes.length >= 13, 'must have at least 13 operator classes');
  assert.ok(classes.includes('attention'));
  assert.ok(classes.includes('projection'));
  assert.ok(classes.includes('logits'));

  const attnF16 = getDriftTolerance('attention', 'f16');
  assert.ok(attnF16 !== null);
  assert.equal(attnF16.maxAbsDiff, 1e-3);
  assert.equal(attnF16.maxRelDiff, 1e-2);
  assert.equal(attnF16.propagationWeight, 2.0);

  const projF32 = getDriftTolerance('projection', 'f32');
  assert.ok(projF32 !== null);
  assert.equal(projF32.maxAbsDiff, 1e-5);

  const embedQ4k = getDriftTolerance('embedding', 'q4k');
  assert.ok(embedQ4k !== null);
  assert.equal(embedQ4k.maxAbsDiff, 5e-2);

  assert.equal(getDriftTolerance(null, 'f32'), null);
  assert.equal(getDriftTolerance('nonexistent', 'f32'), null);

  assert.equal(getDriftPolicyId('attention'), 'attention');
  assert.equal(getDriftPolicyId('nonexistent'), null);
  assert.equal(getDriftPolicyId(null), null);

  const withinBudget = checkDrift('normalization', 'f32', { maxAbsDiff: 1e-7, maxRelDiff: 1e-6 });
  assert.equal(withinBudget.withinBudget, true);
  assert.equal(withinBudget.reason, 'within_tolerance');

  const absExceeded = checkDrift('normalization', 'f32', { maxAbsDiff: 1, maxRelDiff: 0 });
  assert.equal(absExceeded.withinBudget, false);
  assert.equal(absExceeded.reason, 'abs_exceeded');

  const bothExceeded = checkDrift('projection', 'f32', { maxAbsDiff: 1, maxRelDiff: 1 });
  assert.equal(bothExceeded.withinBudget, false);
  assert.equal(bothExceeded.reason, 'both_exceeded');

  const noPolicy = checkDrift(null, 'f32', { maxAbsDiff: 999 });
  assert.equal(noPolicy.withinBudget, true);
  assert.equal(noPolicy.reason, 'no_policy');

  const propOk = checkPropagationBound('f32', [
    { maxAbsDiff: 1e-7, propagationWeight: 1.0 },
    { maxAbsDiff: 1e-7, propagationWeight: 1.0 },
  ]);
  assert.equal(propOk.withinBound, true);

  const propBad = checkPropagationBound('f32', [
    { maxAbsDiff: 1e-2, propagationWeight: 1.0 },
    { maxAbsDiff: 1e-2, propagationWeight: 1.0 },
  ]);
  assert.equal(propBad.withinBound, false);
  assert.equal(propBad.reason, 'accumulated_drift_exceeded');

  const propEmpty = checkPropagationBound('f32', []);
  assert.equal(propEmpty.withinBound, true);
  assert.equal(propEmpty.reason, 'no_data');

  console.log('  drift-policy: ok');
}

// ============================================================================
// 5. Capture Policy
// ============================================================================
{
  const {
    CAPTURE_LEVELS,
    resolveCapturePolicy,
    escalateCaptureLevel,
    buildCaptureArtifact,
    createEscalationPolicy,
    createDefaultCaptureConfig,
    validateCaptureConfig,
  } = await import('../../src/debug/capture-policy.js');

  assert.equal(CAPTURE_LEVELS.NONE, 'none');
  assert.equal(CAPTURE_LEVELS.METADATA, 'metadata');
  assert.equal(CAPTURE_LEVELS.SLICE, 'slice');
  assert.equal(CAPTURE_LEVELS.FULL, 'full');

  assert.equal(resolveCapturePolicy('any.op', null), 'none');

  const config = createDefaultCaptureConfig();
  assert.equal(config.enabled, false);
  assert.equal(resolveCapturePolicy('any.op', config), 'none');

  config.enabled = true;
  config.defaultLevel = 'metadata';
  assert.equal(resolveCapturePolicy('any.op', config), 'metadata');

  config.targetOpIds = ['layer.5.attn.q_proj'];
  config.targetLevel = 'full';
  assert.equal(resolveCapturePolicy('layer.5.attn.q_proj', config), 'full');
  assert.equal(resolveCapturePolicy('layer.0.ffn.out', config), 'metadata');

  const config2 = createDefaultCaptureConfig();
  config2.enabled = true;
  config2.targetLayers = [3, 7];
  config2.targetLevel = 'slice';
  assert.equal(resolveCapturePolicy('layer.3.attn.out', config2), 'slice');
  assert.equal(resolveCapturePolicy('layer.5.attn.out', config2), 'none');

  assert.equal(escalateCaptureLevel('none', 'metadata'), 'metadata');
  assert.equal(escalateCaptureLevel('slice', 'metadata'), 'slice');
  assert.equal(escalateCaptureLevel('metadata', 'full'), 'full');
  assert.equal(escalateCaptureLevel('full', 'none'), 'full');

  const metaArtifact = buildCaptureArtifact('op1', 'metadata', new Float32Array([1, 2, 3, 4]), { shape: [4] });
  assert.equal(metaArtifact.opId, 'op1');
  assert.equal(metaArtifact.level, 'metadata');
  assert.ok(metaArtifact.stats !== null);
  assert.equal(metaArtifact.stats.elementCount, 4);
  assert.equal(metaArtifact.stats.min, 1);
  assert.equal(metaArtifact.stats.max, 4);
  assert.equal(metaArtifact.sample, undefined);
  assert.equal(metaArtifact.data, undefined);

  const sliceArtifact = buildCaptureArtifact('op2', 'slice', new Float32Array([10, 20, 30]), { sampleCount: 2 });
  assert.ok(sliceArtifact.sample !== null);
  assert.ok(sliceArtifact.sample.length <= 2);
  assert.equal(sliceArtifact.data, undefined);

  const fullArtifact = buildCaptureArtifact('op3', 'full', new Float32Array([1, 2]));
  assert.deepEqual(fullArtifact.data, [1, 2]);

  const noneArtifact = buildCaptureArtifact('op4', 'none', null);
  assert.equal(noneArtifact.stats, undefined);

  const nanArtifact = buildCaptureArtifact('op5', 'metadata', new Float32Array([NaN, 1, Infinity]));
  assert.equal(nanArtifact.stats.nanCount, 1);
  assert.equal(nanArtifact.stats.infCount, 1);

  const esc = createEscalationPolicy({ windowBefore: 2, windowAfter: 1 });
  assert.equal(esc.resolveForIndex(4, 6), 'full');
  assert.equal(esc.resolveForIndex(5, 6), 'full');
  assert.equal(esc.resolveForIndex(6, 6), 'full');
  assert.equal(esc.resolveForIndex(7, 6), 'full');
  assert.equal(esc.resolveForIndex(8, 6), 'metadata');
  assert.equal(esc.resolveForIndex(0, 6), 'metadata');
  assert.equal(esc.resolveForIndex(3, null), 'metadata');

  assert.ok(validateCaptureConfig(config));
  assert.throws(() => validateCaptureConfig(null), /config must be an object/);
  assert.throws(
    () => validateCaptureConfig({ defaultLevel: 'invalid' }),
    /Invalid defaultLevel/
  );

  console.log('  capture-policy: ok');
}

// ============================================================================
// 6. Failure Reduction
// ============================================================================
{
  const {
    REDUCER_MODES,
    createReductionConfig,
    computePromptReductionSteps,
    computeTokenReductionPlan,
    createGraphSlice,
    createReductionReport,
  } = await import('../../src/inference/pipelines/text/failure-reduction.js');

  assert.equal(REDUCER_MODES.PROMPT, 'prompt');
  assert.equal(REDUCER_MODES.TOKENS, 'tokens');
  assert.equal(REDUCER_MODES.GRAPH, 'graph');

  const defaultConfig = createReductionConfig();
  assert.equal(defaultConfig.mode, null);
  assert.equal(defaultConfig.enabled, false);
  assert.equal(defaultConfig.prompt.minLength, 1);
  assert.equal(defaultConfig.prompt.strategy, 'binary_search');
  assert.equal(defaultConfig.tokens.maxTokens, 1);

  const promptConfig = createReductionConfig({ mode: 'prompt', enabled: true, promptMinLength: 2 });
  const steps = computePromptReductionSteps([1, 2, 3, 4, 5, 6, 7, 8], promptConfig);
  assert.ok(steps.length > 0, 'must produce at least one step');
  assert.equal(steps[steps.length - 1].length, 2, 'last step must be minLength');
  for (let i = 1; i < steps.length; i++) {
    assert.ok(steps[i].length <= steps[i - 1].length, 'steps must be non-increasing');
  }

  const linearConfig = createReductionConfig({ promptMinLength: 2, promptStrategy: 'linear' });
  const linearSteps = computePromptReductionSteps([1, 2, 3, 4, 5, 6, 7, 8], linearConfig);
  assert.ok(linearSteps.length > 0);
  assert.equal(linearSteps[linearSteps.length - 1].length, 2);

  const tinyPrompt = computePromptReductionSteps([1], createReductionConfig({ promptMinLength: 1 }));
  assert.equal(tinyPrompt.length, 1);
  assert.deepEqual(tinyPrompt[0], [1]);

  const tokenConfig = createReductionConfig({ maxTokens: 1, tokenStep: 3 });
  const tokenPlan = computeTokenReductionPlan(10, tokenConfig);
  assert.ok(tokenPlan.length > 0);
  assert.equal(tokenPlan[0], 10);
  assert.equal(tokenPlan[tokenPlan.length - 1], 1);
  for (let i = 1; i < tokenPlan.length; i++) {
    assert.ok(tokenPlan[i] <= tokenPlan[i - 1], 'plan must be non-increasing');
  }

  const sliceConfig = createReductionConfig({ startLayer: 5, endLayer: 10 });
  const slice = createGraphSlice(sliceConfig);
  assert.equal(slice.shouldProcessLayer(3), false);
  assert.equal(slice.shouldProcessLayer(5), true);
  assert.equal(slice.shouldProcessLayer(7), true);
  assert.equal(slice.shouldProcessLayer(10), true);
  assert.equal(slice.shouldProcessLayer(11), false);
  assert.equal(slice.isSliced(), true);

  const opSlice = createGraphSlice(createReductionConfig({
    targetOpIds: ['layer.5.attn.q_proj'],
  }));
  assert.equal(opSlice.shouldProcessOp('layer.5.attn.q_proj', 'attn.q_proj'), true);
  assert.equal(opSlice.shouldProcessOp('layer.5.ffn.out', 'ffn.out'), false);
  assert.equal(opSlice.isSliced(), true);

  const noSlice = createGraphSlice(createReductionConfig());
  assert.equal(noSlice.shouldProcessLayer(0), true);
  assert.equal(noSlice.shouldProcessLayer(999), true);
  assert.equal(noSlice.isSliced(), false);

  const report = createReductionReport({
    mode: 'prompt',
    originalSize: 100,
    reducedSize: 10,
    stepsAttempted: 5,
    divergenceReproduced: true,
    divergenceOpId: 'layer.3.attn.softmax',
    minimalPromptLength: 10,
  });
  assert.equal(report.mode, 'prompt');
  assert.equal(report.originalSize, 100);
  assert.equal(report.reducedSize, 10);
  assert.equal(report.divergenceReproduced, true);
  assert.equal(report.divergenceOpId, 'layer.3.attn.softmax');
  assert.equal(report.minimalPromptLength, 10);

  console.log('  failure-reduction: ok');
}

console.log('operator-diffing.test: ok');
