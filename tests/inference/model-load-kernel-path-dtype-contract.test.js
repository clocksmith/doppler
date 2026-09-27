import assert from 'node:assert/strict';
import { createKernelRegistry, enterKernelRegistry, setKernelValidator } from '../../src/gpu/kernels/kernel-configs.js';
import { listPrewarmKernels } from '../../src/gpu/kernels/kernel-prewarm.js';

const { resolveKernelPathState } = await import('../../src/inference/pipelines/text/model-load.js');

function createKernelPath(activationDtype = 'f32') {
  return {
    id: `inline-${activationDtype}`,
    name: `Inline ${activationDtype.toUpperCase()}`,
    activationDtype,
    outputDtype: activationDtype,
    kvDtype: activationDtype,
    decode: {
      steps: [
        { op: 'attention', kernel: 'attention_streaming_f16kv.wgsl' },
      ],
    },
    prefill: {
      steps: [
        { op: 'attention', kernel: 'attention_streaming_f16kv.wgsl' },
      ],
    },
  };
}

function createRuntimeConfig(dtype) {
  return {
    inference: {
      kernelPath: null,
      compute: {
        activationDtype: dtype,
      },
      session: {
        compute: {
          defaults: {
            activationDtype: dtype,
            outputDtype: dtype,
          },
        },
        kvcache: {
          kvDtype: dtype,
        },
      },
    },
  };
}

function createManifest(compute = 'f32') {
  return {
    modelId: 'model-load-kernel-path-dtype-contract',
    quantizationInfo: {
      compute,
    },
    inference: {},
  };
}

{
  assert.throws(
    () => resolveKernelPathState({
      manifest: createManifest('f32'),
      runtimeConfig: createRuntimeConfig('f16'),
      modelConfig: {
        kernelPath: createKernelPath('f32'),
      },
    }),
    /Runtime dtype auto-rewrites are not allowed/
  );
}

{
  const runtimeConfig = createRuntimeConfig('f32');
  const result = resolveKernelPathState({
    manifest: createManifest('f32'),
    runtimeConfig,
    modelConfig: {
      kernelPath: createKernelPath('f32'),
    },
  });

  assert.strictEqual(result.runtimeConfig, runtimeConfig);
  assert.equal(result.resolvedKernelPath?.activationDtype, 'f32');
  assert.equal(result.kernelPathSource, 'model');
}

const scopedShader = 'scoped-preflight.wgsl';
function scopedRegistry(variant, requires) {
  return createKernelRegistry({ extensions: { gelu: { variants: {
    [variant]: { wgsl: scopedShader, entryPoint: 'main', workgroup: [256, 1, 1], requires },
  } } } });
}
const registryA = scopedRegistry('scoped_a', ['shader-f16']);
const registryB = scopedRegistry('scoped_b', ['subgroups']);
const scopedPath = {
  ...createKernelPath(),
  decode: { steps: [{ op: 'activation', kernel: scopedShader, entry: 'main' }] },
  prefill: { steps: [] },
};
const f16Missing = { hasF16: false, hasSubgroups: true, wgslLanguageFeatures: [] };
const subgroupsMissing = { hasF16: true, hasSubgroups: false, wgslLanguageFeatures: [] };
function checkScopedPreflight(registry, capabilities) {
  const restore = enterKernelRegistry(registry);
  try {
    return resolveKernelPathState({
      manifest: createManifest(),
      runtimeConfig: createRuntimeConfig('f32'),
      modelConfig: { kernelPath: scopedPath },
      kernelCapabilities: capabilities,
    });
  } finally { restore(); }
}
function prewarmVariants(registry, capabilities) {
  return listPrewarmKernels(registry, capabilities)
    .flatMap(([operation, variants]) => variants.map(([variant]) => `${operation}/${variant}`));
}
assert.throws(() => checkScopedPreflight(registryA, f16Missing), /scoped-preflight.wgsl.*shader-f16/);
assert.doesNotThrow(() => checkScopedPreflight(registryB, f16Missing));
assert.doesNotThrow(() => checkScopedPreflight(registryA, subgroupsMissing));
assert.throws(() => checkScopedPreflight(registryB, subgroupsMissing), /scoped-preflight.wgsl.*subgroups/);
assert.equal(prewarmVariants(registryA, f16Missing).includes('gelu/scoped_a'), false);
assert.equal(prewarmVariants(registryB, f16Missing).includes('gelu/scoped_b'), true);
assert.equal(prewarmVariants(registryA, subgroupsMissing).includes('gelu/scoped_a'), true);
assert.equal(prewarmVariants(registryB, subgroupsMissing).includes('gelu/scoped_b'), false);
assert.equal(prewarmVariants(registryA, subgroupsMissing).includes('gelu/scoped_b'), false);
assert.equal(prewarmVariants(registryB, f16Missing).includes('gelu/scoped_a'), false);
setKernelValidator('gelu', 'gelu', () => {});
assert.throws(() => checkScopedPreflight(registryA, f16Missing), /scoped-preflight.wgsl.*shader-f16/);
assert.equal(prewarmVariants(registryB, f16Missing).includes('gelu/scoped_b'), true);

console.log('model-load-kernel-path-dtype-contract.test: ok');
