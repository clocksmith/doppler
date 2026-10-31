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
// Inline-path admission must read the session contract without inferring it from the shader.
for (const dtype of ['f16', 'f32']) {
  const runtimeConfig = createRuntimeConfig(dtype);
  runtimeConfig.inference.kernelPath = createKernelPath(dtype);
  delete runtimeConfig.inference.compute;
  const original = structuredClone(runtimeConfig);
  const resolve = (config = runtimeConfig, runtimeOverrides = null) => resolveKernelPathState({
    manifest: createManifest(dtype), runtimeConfig: config, runtimeOverrides,
    modelConfig: { kernelPath: null },
  });
  const result = resolve();
  assert.equal(result.runtimeConfig.inference.compute.activationDtype, dtype);
  assert.deepEqual(runtimeConfig, original, 'compatibility projection must not mutate the caller');
  assert.deepEqual(result.runtimeConfig.inference.session, original.inference.session);

  const other = dtype === 'f32' ? 'f16' : 'f32';
  const conflict = structuredClone(runtimeConfig);
  conflict.inference.compute = { activationDtype: other };
  assert.throws(() => resolve(conflict), /conflicts with legacy/);
  assert.throws(() => resolve(runtimeConfig, conflict), /conflicts with legacy/);
  assert.throws(() => resolve(runtimeConfig, { inference: { compute: { activationDtype: other } } }),
    /requires explicit matching runtime dtypes/);

  for (const invalid of [null, '', 'bf16']) {
    const bad = structuredClone(runtimeConfig);
    bad.inference.session.compute.defaults.activationDtype = invalid;
    assert.throws(() => resolve(bad), /activationDtype.*(required|must be)/);
  }
  const missing = structuredClone(runtimeConfig);
  delete missing.inference.session.compute.defaults.activationDtype;
  assert.throws(() => resolve(missing), /requires explicit matching runtime dtypes/);
  missing.inference.compute = { activationDtype: dtype };
  assert.strictEqual(resolve(missing).runtimeConfig, missing, 'legacy-only compatibility remains');
  missing.inference.compute.activationDtype = dtype === 'f32' ? 'float32' : 'fp16';
  assert.strictEqual(resolve(missing).runtimeConfig, missing, 'legacy dtype aliases retain their existing rule contract');

  const mismatch = structuredClone(runtimeConfig);
  mismatch.inference.session.compute.defaults.activationDtype = other;
  assert.throws(() => resolve(mismatch), /requires explicit matching runtime dtypes/);
}

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
