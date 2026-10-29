import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');
const { getLayerSteps } = await import('../../src/config/kernel-path-loader.js');

// Contract: the Gemma 4 31B af16 manifest variant is an all-f16 activation lane.
// Under the probe profile every activation-carrying step leaves the af32 kernel
// set, no f32 precision boundary or mixed f16kv attention remains, and the af32
// variant fails closed when asked for f16. Expected values are derived from the
// capability rule, the profile and an af32 baseline compile, not copied kernels.

const AF32_MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af16';

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(new URL(relativePath, import.meta.url), 'utf8'));
}

const af32Config = readJson(`../../src/config/conversion/gemma4/${AF32_MODEL_ID}.json`);
const af16Config = readJson(`../../src/config/conversion/gemma4/${AF16_MODEL_ID}.json`);
const profile = readJson('../../src/config/runtime/profiles/gemma4-31b-f16-activations-probe.json');
const rules = readJson('../../src/rules/inference/capability-transforms.rules.json').capabilityTransforms;

const laneRule = rules.find((rule) => rule.match?.modelId === AF16_MODEL_ID);
const guardRule = rules.find((rule) => rule.match?.modelId === AF32_MODEL_ID);
assert.ok(laneRule, 'af16 variant must have an explicit capability lane rule');
assert.equal(laneRule.kind, 'explicit-lane');
assert.equal(guardRule?.kind, 'lane-mismatch-guard', 'af32 variant must fail closed for f16 requests');

// Variant identity: shared weights, distinct manifest variant, profile bound to af16.
assert.equal(af16Config.output.modelBaseId, AF16_MODEL_ID);
assert.equal(
  af16Config.manifest?.artifactIdentity?.weightCapsuleId,
  af32Config.manifest?.artifactIdentity?.weightCapsuleId
);
assert.notEqual(
  af16Config.manifest?.artifactIdentity?.manifestVariantId,
  af32Config.manifest?.artifactIdentity?.manifestVariantId
);
assert.equal(profile.model, AF16_MODEL_ID);

// The profile requests exactly the dtypes the lane rule matches on.
const requested = profile.runtime.inference.session.compute.defaults;
for (const field of ['activationDtype', 'mathDtype', 'accumDtype']) {
  assert.equal(requested[field], laneRule.match[field], `profile ${field} must match the lane rule`);
}
assert.equal(profile.runtime.inference.compute.activationDtype, requested.activationDtype);

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
    kernelPathPolicy: profile.runtime.inference.kernelPathPolicy,
    capabilities: { hasSubgroups: true, hasF16: true, maxWorkgroupStorageSize: 32768 },
    platform: { id: 'rdna3', vendor: 'amd', architecture: 'rdna-3' },
  });
}

const compiled = compile(af16Config, profile.runtime.inference.session);
const baseline = compile(af32Config, undefined);

assert.deepEqual(compiled.appliedTransforms, laneRule.transforms);
assert.deepEqual(baseline.appliedTransforms, []);
assert.throws(
  () => compile(af32Config, profile.runtime.inference.session),
  /lane mismatch/,
  'af32 manifest must not dispatch the f16 lane'
);

const kernelPath = compiled.runtimeInferencePatch.kernelPath;
assert.ok(kernelPath, 'Gemma 4 31B f16 compile should build an inline kernel path');
assert.equal(compiled.runtimeInferencePatch.compute?.activationDtype, requested.activationDtype);
for (const field of ['activationDtype', 'mathDtype', 'accumDtype', 'outputDtype']) {
  assert.equal(compiled.runtimeInferencePatch.session?.compute?.defaults?.[field], requested[field], field);
}
assert.equal(kernelPath.activationDtype, requested.activationDtype);
assert.equal(kernelPath.kvDtype, profile.runtime.inference.session.kvcache.kvDtype);

function assertAllF16(laneSteps, baseSteps, where) {
  assert.deepEqual(laneSteps.map((step) => step.op), baseSteps.map((step) => step.op),
    `${where}: op order must match the af32 graph`);
  for (const [index, step] of laneSteps.entries()) {
    assert.notEqual(step.kernel, baseSteps[index].kernel,
      `${where} ${step.op}: must leave the af32 kernel`);
    for (const [key, value] of Object.entries(step.precision ?? {})) {
      assert.notEqual(value, 'f32', `${where} ${step.op}: precision.${key} must not be f32`);
    }
  }
}

const layerTypes = af16Config.inference.layerPattern.layerTypes;
const representativeLayers = [...new Set(layerTypes)].map((type) => layerTypes.indexOf(type));
assert.ok(representativeLayers.length >= 2, 'expected sliding and full attention layers');

const baselinePath = baseline.runtimeInferencePatch.kernelPath;
assertAllF16(kernelPath.preLayer, baselinePath.preLayer, 'preLayer');
for (const layer of representativeLayers) {
  for (const phase of ['decode', 'prefill']) {
    assertAllF16(
      getLayerSteps(kernelPath, layer, phase),
      getLayerSteps(baselinePath, layer, phase),
      `layer ${layer} (${layerTypes[layer]}) ${phase}`
    );
  }
}
assertAllF16(kernelPath.postLayer, baselinePath.postLayer, 'postLayer');

for (const step of compiled.resolvedSteps.all) {
  assert.ok(
    !String(step.kernel).includes('_f16kv'),
    `Gemma 4 31B f16 lane should not keep f16kv attention in resolved step ${step.id}`
  );
}

console.log('gemma4-31b-f16-activations.test: ok');
