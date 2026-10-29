import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');
const { getLayerSteps } = await import('../../src/config/kernel-path-loader.js');

// Contract: the Gemma 4 12B af16 manifest variant runs an f16 residual stream
// with stable boundaries. Attention keeps f32 queries over an f16 KV cache, the
// Q/K/V projections that feed it stay f32, the post-layer tail is unchanged
// from the af32 graph, and the residual-writing projections run f16. The af32
// variant fails closed for f16 requests. Expectations are derived from the
// capability rule and an af32 baseline compile, not copied kernel names.

const AF32_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af16';
const ATTENTION_INPUT_PROJECTIONS = ['q_proj', 'k_proj', 'v_proj'];
const RESIDUAL_WRITING_PROJECTIONS = ['o_proj', 'down_proj'];

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(new URL(relativePath, import.meta.url), 'utf8'));
}

const af32Config = readJson(`../../src/config/conversion/gemma4/${AF32_MODEL_ID}.json`);
const af16Config = readJson(`../../src/config/conversion/gemma4/${AF16_MODEL_ID}.json`);
const rules = readJson('../../src/rules/inference/capability-transforms.rules.json').capabilityTransforms;

const laneRule = rules.find((rule) => rule.match?.modelId === AF16_MODEL_ID);
const guardRule = rules.find((rule) => rule.match?.modelId === AF32_MODEL_ID);
assert.ok(laneRule, 'af16 variant must have an explicit capability lane rule');
assert.equal(laneRule.kind, 'explicit-lane');
assert.equal(guardRule?.kind, 'lane-mismatch-guard', 'af32 variant must fail closed for f16 requests');

// Variant identity: shared weights, distinct manifest variant, f16 compute precision.
assert.equal(af16Config.output.modelBaseId, AF16_MODEL_ID);
assert.equal(
  af16Config.manifest?.artifactIdentity?.weightCapsuleId,
  af32Config.manifest?.artifactIdentity?.weightCapsuleId
);
assert.notEqual(
  af16Config.manifest?.artifactIdentity?.manifestVariantId,
  af32Config.manifest?.artifactIdentity?.manifestVariantId
);
assert.equal(af16Config.quantization.computePrecision, laneRule.match.activationDtype);

const requestedDefaults = {
  activationDtype: laneRule.match.activationDtype,
  mathDtype: laneRule.match.mathDtype,
  accumDtype: laneRule.match.accumDtype,
  outputDtype: laneRule.match.activationDtype,
};

function f16Session(conversionConfig) {
  return {
    ...conversionConfig.session,
    compute: { ...conversionConfig.session.compute, defaults: requestedDefaults },
    kvcache: { ...conversionConfig.session.kvcache, kvDtype: 'f16' },
  };
}

function compile(conversionConfig, runtimeSession) {
  return compileExecutionV1({
    manifestInference: {
      schema: 'doppler.execution/v1',
      session: conversionConfig.session,
      layerPattern: conversionConfig.inference.layerPattern,
      execution: conversionConfig.execution,
    },
    modelId: conversionConfig.output.modelBaseId,
    numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
    headDim: 512,
    runtimeSession,
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['manifest'],
      allowSources: ['manifest'],
      onIncompatible: 'remap',
    },
    capabilities: { hasSubgroups: true, hasF16: true, maxWorkgroupStorageSize: 32768 },
    platform: { id: 'rdna3', vendor: 'amd', architecture: 'rdna-3' },
  });
}

const compiled = compile(af16Config, f16Session(af16Config));
const baseline = compile(af32Config, undefined);

assert.deepEqual(compiled.appliedTransforms, laneRule.transforms);
assert.deepEqual(baseline.appliedTransforms, []);
assert.throws(
  () => compile(af32Config, f16Session(af32Config)),
  /lane mismatch/,
  'af32 manifest must not dispatch the f16 lane'
);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'Gemma 4 12B f16 compile should build an inline kernel path');
assert.equal(compiled.runtimeInferencePatch.compute?.activationDtype, requestedDefaults.activationDtype);
assert.deepEqual(compiled.runtimeInferencePatch.session?.compute?.defaults, requestedDefaults);
assert.equal(kernelPath.activationDtype, requestedDefaults.activationDtype);
assert.equal(kernelPath.kvDtype, 'f16');

const baselinePath = baseline.runtimeInferencePatch.kernelPath;

const embed = kernelPath.preLayer.find((step) => step.op === 'embed');
assert.notEqual(embed?.kernel, baselinePath.preLayer.find((step) => step.op === 'embed')?.kernel);
assert.equal(embed?.precision?.outputDtype, 'f16', 'embedding must feed the f16 residual stream');

const layerTypes = af16Config.inference.layerPattern.layerTypes;
const representativeLayers = [...new Set(layerTypes)].map((type) => layerTypes.indexOf(type));
assert.ok(representativeLayers.length >= 2, 'expected sliding and full attention layers');

for (const layer of representativeLayers) {
  for (const phase of ['decode', 'prefill']) {
    const where = `layer ${layer} (${layerTypes[layer]}) ${phase}`;
    const laneSteps = getLayerSteps(kernelPath, layer, phase);
    const baseSteps = getLayerSteps(baselinePath, layer, phase);
    assert.deepEqual(laneSteps.map((step) => step.op), baseSteps.map((step) => step.op),
      `${where}: op order must match the af32 graph`);
    const lane = (op) => laneSteps.find((step) => step.op === op);
    const base = (op) => baseSteps.find((step) => step.op === op);

    // Stable attention boundary: same fixed-head f16kv kernel, f32 queries.
    assert.equal(lane('attention').kernel, base('attention').kernel, `${where} attention`);
    assert.equal(lane('attention').precision?.activationDtype, 'f32', `${where} attention`);
    assert.equal(lane('attention').precision?.kvDtype, 'f16', `${where} attention`);

    for (const op of ATTENTION_INPUT_PROJECTIONS) {
      assert.equal(lane(op).kernel, base(op).kernel, `${where} ${op}`);
      assert.deepEqual(lane(op).precision, { inputDtype: 'f32', outputDtype: 'f32' }, `${where} ${op}`);
    }
    for (const op of RESIDUAL_WRITING_PROJECTIONS) {
      assert.notEqual(lane(op).kernel, base(op).kernel, `${where} ${op} must use an f16-activation kernel`);
      assert.deepEqual(lane(op).precision, { inputDtype: 'f16', outputDtype: 'f16' }, `${where} ${op}`);
    }
    for (const op of ['attn_residual', 'ffn_residual']) {
      assert.notEqual(lane(op).kernel, base(op).kernel, `${where} ${op} must operate on the f16 residual stream`);
    }
  }
}

assert.deepEqual(
  kernelPath.postLayer.map((step) => [step.op, step.kernel, step.entry, step.precision ?? null]),
  baselinePath.postLayer.map((step) => [step.op, step.kernel, step.entry, step.precision ?? null]),
  'Gemma 4 12B f16 lane keeps the af32 post-layer tail'
);

console.log('gemma4-12b-f16-activations.test: ok');
