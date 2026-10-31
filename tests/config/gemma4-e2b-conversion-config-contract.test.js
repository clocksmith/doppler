import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getLayerSteps } from '../../src/config/kernel-path-loader.js';

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, '..', '..');

const configPath = path.join(
  repoRoot,
  'src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32.json'
);
const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
const manifestInference = {
  schema: 'doppler.execution/v1',
  ...config.inference,
  session: config.session,
  execution: config.execution,
};
const modelHeadDim = 256;

const f16Primary = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
    maxWorkgroupStorageSize: 32768,
  },
  platform: {
    id: 'test-f16',
    vendor: 'test',
    architecture: 'test',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.deepEqual(f16Primary.appliedTransforms, []);
assert.equal(f16Primary.session.compute.defaults.activationDtype, 'f32');
assert.equal(f16Primary.session.kvcache.kvDtype, 'f16');
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.kernel,
  'attention_decode_online_f16kv.wgsl'
);
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  f16Primary.runtimeInferencePatch?.kernelPath?.prefill?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f16'
);
assert.equal(
  getLayerSteps(f16Primary.runtimeInferencePatch?.kernelPath, 0, 'prefill')
    .find((step) => step.op === 'attention')?.kernel,
  'attention_head256_f16kv.wgsl'
);
assert.equal(
  getLayerSteps(f16Primary.runtimeInferencePatch?.kernelPath, 4, 'prefill')
    .find((step) => step.op === 'attention')?.kernel,
  'attention_head512_f16kv.wgsl'
);

const appleRetainDisabled = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  runtimeSession: {
    retainQ4KMaterialization: true,
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'apple-m3',
    vendor: 'apple',
    architecture: 'metal-3',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.deepEqual(appleRetainDisabled.appliedTransforms, ['disableRetainQ4KMaterialization']);
assert.equal(appleRetainDisabled.session.retainQ4KMaterialization, false);
assert.equal(
  appleRetainDisabled.runtimeInferencePatch.session?.retainQ4KMaterialization,
  false
);

const amdRetainAllowed = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  runtimeSession: {
    retainQ4KMaterialization: true,
  },
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'amd-rdna3',
    vendor: 'amd',
    architecture: 'rdna3',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.deepEqual(amdRetainAllowed.appliedTransforms, []);
assert.equal(amdRetainAllowed.session.retainQ4KMaterialization, true);
assert.equal(
  amdRetainAllowed.runtimeInferencePatch.session?.retainQ4KMaterialization,
  true
);

assert.throws(
  () => compileExecutionV1({
    manifestInference,
    modelId: config.output.modelBaseId,
    numLayers: 35,
    headDim: modelHeadDim,
    runtimeSession: {
      ...config.session,
      compute: {
        ...config.session.compute,
        defaults: {
          ...config.session.compute.defaults,
          activationDtype: 'f16',
          mathDtype: 'f16',
          outputDtype: 'f16',
        },
      },
    },
    capabilities: {
      hasSubgroups: true,
      hasF16: true,
      hasSubgroupsF16: true,
    },
    platform: {
      id: 'test-f16-runtime',
      vendor: 'test',
      architecture: 'test',
    },
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['manifest', 'model', 'config'],
      onIncompatible: 'remap',
    },
  }),
  /attention_head(?:256|512)_f16kv\.wgsl" requires activationDtype="f32" and kvcache\.kvDtype="f16"/
);

const finitenessFallback = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    hasSubgroupsF16: true,
  },
  platform: {
    id: 'test-f16-fallback',
    vendor: 'test',
    architecture: 'test',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.equal(finitenessFallback.fallbackKernelPath, null);

const widenedFallback = compileExecutionV1({
  manifestInference,
  modelId: config.output.modelBaseId,
  numLayers: 35,
  headDim: modelHeadDim,
  capabilities: {
    hasSubgroups: true,
    hasF16: false,
    hasSubgroupsF16: false,
  },
  platform: {
    id: 'test-f32-fallback',
    vendor: 'test',
    architecture: 'test',
  },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

assert.deepEqual(widenedFallback.appliedTransforms, ['widenToF32Activations']);
assert.equal(widenedFallback.session.compute.defaults.activationDtype, 'f32');
assert.equal(widenedFallback.session.kvcache.kvDtype, 'f32');
assert.equal(
  widenedFallback.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.kernel,
  'attention_decode_online.wgsl'
);
assert.equal(
  widenedFallback.runtimeInferencePatch?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.precision?.kvDtype,
  'f32'
);

console.log('gemma4-e2b-conversion-config-contract.test: ok');
