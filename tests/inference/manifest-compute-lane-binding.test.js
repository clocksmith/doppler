import assert from 'node:assert/strict';

const { assertManifestComputeLaneBinding } = await import(
  '../../src/inference/pipelines/text/model-load.js'
);

function makeRuntimeConfig({ activationDtype, mathDtype, accumDtype, kvDtype }) {
  return {
    inference: {
      session: {
        compute: {
          defaults: {
            ...(activationDtype !== undefined ? { activationDtype } : {}),
            ...(mathDtype !== undefined ? { mathDtype } : {}),
            ...(accumDtype !== undefined ? { accumDtype } : {}),
          },
        },
        kvcache: {
          ...(kvDtype !== undefined ? { kvDtype } : {}),
        },
      },
    },
  };
}

function makeManifest(compute) {
  return {
    modelId: 'test-manifest-compute-lane',
    quantizationInfo: compute === undefined ? {} : { compute },
  };
}

// f32 manifest, f32 runtime → ok
{
  assertManifestComputeLaneBinding({
    manifest: makeManifest('f32'),
    runtimeConfig: makeRuntimeConfig({
      activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', kvDtype: 'f32',
    }),
  });
}

// f16 manifest, f16 runtime → ok
{
  assertManifestComputeLaneBinding({
    manifest: makeManifest('f16'),
    runtimeConfig: makeRuntimeConfig({
      activationDtype: 'f16', mathDtype: 'f16', accumDtype: 'f16', kvDtype: 'f16',
    }),
  });
}

// f32 manifest, f16 activationDtype → throws
{
  assert.throws(
    () => assertManifestComputeLaneBinding({
      manifest: makeManifest('f32'),
      runtimeConfig: makeRuntimeConfig({
        activationDtype: 'f16', mathDtype: 'f32', accumDtype: 'f32', kvDtype: 'f32',
      }),
    }),
    /quantizationInfo\.compute=f32 but runtime resolved \[session\.compute\.defaults\.activationDtype=f16/,
  );
}

// KV cache dtype is orthogonal to the compute lane — f32 compute paired with
// f16 KV is the conventional Gemma-family layout, and the inverse must also
// pass. The gate must stay silent for any kvDtype value once activation/math/
// accum agree with the manifest's compute lane.
{
  assertManifestComputeLaneBinding({
    manifest: makeManifest('f16'),
    runtimeConfig: makeRuntimeConfig({
      activationDtype: 'f16', mathDtype: 'f16', accumDtype: 'f16', kvDtype: 'f32',
    }),
  });
  assertManifestComputeLaneBinding({
    manifest: makeManifest('f32'),
    runtimeConfig: makeRuntimeConfig({
      activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', kvDtype: 'f16',
    }),
  });
}

// Manifest without quantizationInfo.compute → silent (legacy/vision-only manifests)
{
  assertManifestComputeLaneBinding({
    manifest: makeManifest(undefined),
    runtimeConfig: makeRuntimeConfig({
      activationDtype: 'f16', kvDtype: 'f32',
    }),
  });
}

// Runtime with no resolved dtypes → silent (cannot compare)
{
  assertManifestComputeLaneBinding({
    manifest: makeManifest('f16'),
    runtimeConfig: { inference: { session: { compute: { defaults: {} }, kvcache: {} } } },
  });
}

// bf16 alias normalizes to f16 — manifest declaring f16, runtime declaring bf16 → ok
{
  assertManifestComputeLaneBinding({
    manifest: makeManifest('f16'),
    runtimeConfig: makeRuntimeConfig({
      activationDtype: 'bf16', kvDtype: 'f16',
    }),
  });
}

console.log('manifest-compute-lane-binding.test: ok');
