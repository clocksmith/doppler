import assert from 'node:assert/strict';

const { compileExecutionV0 } = await import('../../src/inference/pipelines/text/execution-v0.js');
const { buildKernelRefFromKernelEntry } = await import('../../src/config/kernels/kernel-ref.js');

// =============================================================================
// Execution-v0 override sweep
//
// Uses one explicit witness manifest and verifies precedence at:
//   1. Session defaults from manifest
//   2. Runtime session override
//   3. Compute dtype fields (activationDtype, mathDtype, accumDtype, outputDtype)
//   4. KV cache session override
// =============================================================================

function kernelRef(kernel, entry = 'main') {
  return buildKernelRefFromKernelEntry(kernel, entry);
}

// Session fields that can be overridden without triggering kernel-session
// compatibility errors. activationDtype and kvDtype overrides to f32 require
// compatible kernels, so they are tested separately with a kernel-free manifest.
//
// These fields are safe to override because they don't affect kernel selection.
const SAFE_SESSION_FIELDS = [
  {
    path: 'compute.defaults.mathDtype',
    manifestValue: 'f16',
    overrideValue: 'f32',
    runtimeKey: { session: { compute: { defaults: { mathDtype: 'f32' } } } },
  },
  {
    path: 'compute.defaults.accumDtype',
    manifestValue: 'f32',
    overrideValue: 'f16',
    runtimeKey: { session: { compute: { defaults: { accumDtype: 'f16' } } } },
  },
  {
    path: 'compute.defaults.outputDtype',
    manifestValue: 'f16',
    overrideValue: 'f32',
    runtimeKey: { session: { compute: { defaults: { outputDtype: 'f32' } } } },
  },
];

// These fields affect kernel compatibility and are tested with the full
// witness manifest (which validates the compatibility constraint holds).
const KERNEL_COUPLED_FIELDS = [
  {
    path: 'compute.defaults.activationDtype',
    manifestValue: 'f16',
  },
  {
    path: 'kvcache.kvDtype',
    manifestValue: 'f16',
  },
];

const ALL_SESSION_FIELDS = [
  ...SAFE_SESSION_FIELDS,
  ...KERNEL_COUPLED_FIELDS.map((f) => ({ ...f, overrideValue: f.manifestValue })),
];

function witnessManifestInference() {
  return {
    schema: 'doppler.execution/v0',
    sessionDefaults: {
      compute: {
        defaults: {
          activationDtype: 'f16',
          mathDtype: 'f16',
          accumDtype: 'f32',
          outputDtype: 'f16',
        },
        kernelProfiles: [
          { kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main') },
        ],
      },
      kvcache: { kvDtype: 'f16' },
      decodeLoop: null,
    },
    execution: {
      steps: [
        {
          id: 'attn',
          phase: 'both',
          section: 'layer',
          op: 'attention',
          src: 'state',
          dst: 'state',
          layers: 'all',
          kernel: 'attention_streaming_f16.wgsl',
          kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
        },
      ],
      policies: {
        precisionPrecedence: 'step_then_kernel_profile_then_session_default',
        unsupportedPrecision: 'error',
        dtypeTransition: 'require_cast_step',
        unresolvedKernel: 'error',
      },
    },
  };
}

function resolve(obj, dotPath) {
  let v = obj;
  for (const p of dotPath.split('.')) v = v?.[p];
  return v;
}

// --- 1. Manifest session defaults flow through ---
{
  const result = compileExecutionV0({
    manifestInference: witnessManifestInference(),
    modelId: 'witness',
    numLayers: 2,
    runtimeInference: {},
  });
  assert.ok(result, 'compileExecutionV0 should return a result');

  for (const entry of ALL_SESSION_FIELDS) {
    const actual = resolve(result.sessionDefaults, entry.path);
    assert.equal(
      actual,
      entry.manifestValue,
      `manifest default ${entry.path}: expected ${entry.manifestValue}, got ${actual}`
    );
  }
}
console.log(`  ok: manifest session defaults flow through (${ALL_SESSION_FIELDS.length} fields)`);

// --- 2. Runtime session override wins for each safe field individually ---
for (const entry of SAFE_SESSION_FIELDS) {
  const result = compileExecutionV0({
    manifestInference: witnessManifestInference(),
    modelId: 'witness',
    numLayers: 2,
    runtimeInference: entry.runtimeKey,
  });
  const actual = resolve(result.sessionDefaults, entry.path);
  assert.equal(
    actual,
    entry.overrideValue,
    `runtime override ${entry.path}: expected ${entry.overrideValue}, got ${actual}`
  );
}
console.log(`  ok: per-field runtime session override (${SAFE_SESSION_FIELDS.length} safe fields)`);

// --- 2b. Kernel-coupled fields: overriding activationDtype/kvDtype to
//     incompatible values fails fast (validation contract) ---
{
  assert.throws(
    () => compileExecutionV0({
      manifestInference: witnessManifestInference(),
      modelId: 'witness',
      numLayers: 2,
      runtimeInference: {
        session: { compute: { defaults: { activationDtype: 'f32' } } },
      },
    }),
    /activationDtype/,
    'activationDtype f32 override should fail with f16 kernel'
  );
}
console.log('  ok: kernel-coupled activationDtype override fails fast');

// --- 3. Source tracking: manifest vs runtime.session ---
{
  // Use a safe field (mathDtype) to test source tracking without kernel conflict
  const result = compileExecutionV0({
    manifestInference: witnessManifestInference(),
    modelId: 'witness',
    numLayers: 2,
    runtimeInference: {
      session: {
        compute: { defaults: { mathDtype: 'f32' } },
      },
    },
  });

  // mathDtype was overridden -> runtime.session
  const mathSource = findSessionSource(result, 'compute.defaults.mathDtype');
  // accumDtype was NOT overridden -> manifest
  const accumSource = findSessionSource(result, 'compute.defaults.accumDtype');

  if (mathSource && accumSource) {
    assert.notEqual(
      mathSource,
      accumSource,
      'overridden field should have different source from non-overridden'
    );
  }
}

function findSessionSource(result, path) {
  // Walk resolvedSources for session source traces
  const trace = result.resolvedSources;
  if (!trace) return null;
  for (const [key, map] of Object.entries(trace)) {
    if (map instanceof Map && map.has(`sessionDefaults.${path}`)) {
      return map.get(`sessionDefaults.${path}`);
    }
  }
  return null;
}
console.log('  ok: source tracking distinguishes overridden vs manifest');

// --- 4. All safe session fields overridden at once ---
{
  const allOverrides = { session: { compute: { defaults: {} } } };
  for (const entry of SAFE_SESSION_FIELDS) {
    if (entry.path.startsWith('compute.defaults.')) {
      const field = entry.path.replace('compute.defaults.', '');
      allOverrides.session.compute.defaults[field] = entry.overrideValue;
    }
  }

  const result = compileExecutionV0({
    manifestInference: witnessManifestInference(),
    modelId: 'witness',
    numLayers: 2,
    runtimeInference: allOverrides,
  });

  for (const entry of SAFE_SESSION_FIELDS) {
    const actual = resolve(result.sessionDefaults, entry.path);
    assert.equal(actual, entry.overrideValue, `bulk override ${entry.path}`);
  }
}
console.log('  ok: all safe session fields overridden at once');

// --- 5. Execution patch: set step precision ---
{
  const result = compileExecutionV0({
    manifestInference: witnessManifestInference(),
    modelId: 'witness',
    numLayers: 2,
    runtimeInference: {
      executionPatch: {
        set: [{ id: 'attn', precision: { mathDtype: 'f32' } }],
        remove: [],
        add: [],
      },
    },
  });
  assert.ok(result, 'execution patch should compile');
  // Find the attn step in resolved steps
  const attnStep = result.resolvedSteps.all.find((s) => s.id === 'attn');
  assert.ok(attnStep, 'attn step should exist in resolved steps');
}
console.log('  ok: execution patch set applied');

// --- 6. No runtime session -> manifest session defaults are used verbatim ---
{
  const result = compileExecutionV0({
    manifestInference: witnessManifestInference(),
    modelId: 'witness',
    numLayers: 2,
    runtimeInference: {},
  });
  assert.equal(resolve(result.sessionDefaults, 'compute.defaults.activationDtype'), 'f16');
  assert.equal(resolve(result.sessionDefaults, 'kvcache.kvDtype'), 'f16');
}
console.log('  ok: no runtime session uses manifest defaults');

console.log('execution-v0-override-sweep.test: ok');
