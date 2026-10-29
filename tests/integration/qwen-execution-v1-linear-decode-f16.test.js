import assert from 'node:assert/strict';
import fs from 'node:fs';

const { compileExecutionV1 } = await import('../../src/inference/pipelines/text/execution-v1.js');
const { getLayerSteps } = await import('../../src/config/kernel-path-loader.js');

// Contract: the selective f16 lane (useQwenF16PrimaryMatmuls) changes only the
// steps it owns relative to the declared f32 graph. Expected kernels are derived
// from the same config compiled without the lane, not copied from the config.

const LANE_TRANSFORM = 'useQwenF16PrimaryMatmuls';
const F16_DECODE_PROJECTIONS = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj'];
const F32_BOUNDARY_OPS = ['o_proj'];

function loadConversionConfig(relativePath) {
  return JSON.parse(fs.readFileSync(new URL(relativePath, import.meta.url), 'utf8'));
}

function compile(conversionConfig, { activationDtype, hasF16 = true }) {
  return compileExecutionV1({
    manifestInference: {
      schema: 'doppler.execution/v1',
      session: conversionConfig.session,
      layerPattern: conversionConfig.inference.layerPattern,
      execution: conversionConfig.execution,
    },
    modelId: conversionConfig.output.modelBaseId,
    numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
    runtimeCompute: { activationDtype },
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      allowSources: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    },
    capabilities: {
      hasSubgroups: true,
      hasF16,
      maxWorkgroupStorageSize: 32768,
    },
    platform: { id: 'metal', vendor: 'apple', architecture: 'metal-3' },
  });
}

function stepIdentity(step) {
  return {
    kernel: step?.kernel,
    entry: step?.entry,
    constants: step?.constants ?? null,
    precision: step?.precision ?? null,
  };
}

function representativeLayers(conversionConfig) {
  const layerTypes = conversionConfig.inference.layerPattern.layerTypes;
  const layers = [...new Set(layerTypes)].map((type) => layerTypes.indexOf(type));
  assert.ok(layers.length >= 2, 'expected linear and full attention layers in the Qwen 3.5 graph');
  return layers;
}

function assertSelectiveF16Lane(conversionConfig) {
  const label = conversionConfig.output.modelBaseId;
  const lane = compile(conversionConfig, { activationDtype: 'f16' });
  const baseline = compile(conversionConfig, { activationDtype: 'f32' });

  assert.deepEqual(lane.appliedTransforms, [LANE_TRANSFORM], label);
  assert.equal(baseline.appliedTransforms.includes(LANE_TRANSFORM), false, label);
  assert.equal(
    compile(conversionConfig, { activationDtype: 'f16', hasF16: false }).appliedTransforms.includes(LANE_TRANSFORM),
    false,
    `${label}: lane requires shader-f16`
  );

  // The selective lane keeps f32 session activations; only declared steps run f16.
  const kernelPath = lane.runtimeInferencePatch.kernelPath;
  assert.ok(kernelPath, `${label}: execution-v1 compile should build an inline kernel path`);
  assert.equal(lane.runtimeInferencePatch.compute?.activationDtype, 'f32', label);
  assert.equal(kernelPath.activationDtype, 'f32', label);
  assert.equal(kernelPath.kvDtype, conversionConfig.session.kvcache.kvDtype, label);
  assert.equal(lane.fallbackKernelPath, null,
    `${label}: compile must not synthesize a finiteness fallback kernel path`);

  const baselinePath = baseline.runtimeInferencePatch.kernelPath;
  for (const layer of representativeLayers(conversionConfig)) {
    for (const phase of ['decode', 'prefill']) {
      const laneSteps = getLayerSteps(kernelPath, layer, phase);
      const baseSteps = getLayerSteps(baselinePath, layer, phase);
      assert.deepEqual(laneSteps.map((step) => step.op), baseSteps.map((step) => step.op),
        `${label}: layer ${layer} ${phase} op order must not change`);
      for (const [index, step] of laneSteps.entries()) {
        const where = `${label}: layer ${layer} ${phase} ${step.op}`;
        const base = baseSteps[index];
        if (phase === 'decode' && F16_DECODE_PROJECTIONS.includes(step.op)) {
          assert.notEqual(step.kernel, base.kernel, `${where} must use an f16-activation kernel`);
          assert.deepEqual(step.precision, { inputDtype: 'f16', outputDtype: 'f16' }, where);
        } else if (phase === 'decode' && step.op === 'attention') {
          assert.equal(step.precision?.activationDtype, 'f16', where);
          assert.equal(step.precision?.kvDtype, base.precision?.kvDtype, where);
        } else if (F32_BOUNDARY_OPS.includes(step.op)) {
          // The output projection returns to the f32 residual stream.
          assert.equal(step.kernel, base.kernel, where);
          assert.equal(step.entry, base.entry, where);
          assert.deepEqual(step.constants ?? null, base.constants ?? null, where);
          assert.deepEqual(step.precision, { inputDtype: 'f32', outputDtype: 'f32' }, where);
        } else {
          assert.deepEqual(stepIdentity(step), stepIdentity(base), `${where} must be unchanged by the lane`);
        }
      }
    }
  }

  assert.deepEqual(
    kernelPath.postLayer.map(stepIdentity),
    baselinePath.postLayer.map(stepIdentity),
    `${label}: post-layer steps must be unchanged by the lane`
  );
}

for (const configPath of [
  '../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json',
  '../../src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json',
]) {
  assertSelectiveF16Lane(loadConversionConfig(configPath));
}

console.log('qwen-execution-v1-linear-decode-f16.test: ok');
