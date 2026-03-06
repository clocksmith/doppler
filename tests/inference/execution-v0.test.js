import assert from 'node:assert/strict';
import { compileExecutionV0 } from '../../src/inference/pipelines/text/execution-v0.js';
import { buildExecutionV0FromKernelPath } from '../../src/converter/execution-v0-manifest.js';
import { buildKernelRefFromKernelEntry } from '../../src/config/kernels/kernel-ref.js';

const DEFAULT_POLICIES = {
  precisionPrecedence: 'step_then_kernel_profile_then_session_default',
  unsupportedPrecision: 'error',
  dtypeTransition: 'require_cast_step',
  unresolvedKernel: 'error',
};

function kernelRef(kernel, entry = 'main') {
  return buildKernelRefFromKernelEntry(kernel, entry);
}

function sessionDefaultsFor(kernels, activationDtype = 'f16') {
  return {
    compute: {
      defaults: {
        activationDtype,
        mathDtype: activationDtype,
        accumDtype: 'f32',
        outputDtype: activationDtype,
      },
      kernelProfiles: kernels.map(({ kernel, entry }) => ({
        kernelRef: kernelRef(kernel, entry ?? 'main'),
      })),
    },
    kvcache: {
      kvDtype: 'f16',
    },
    decodeLoop: null,
  };
}

{
  const compiled = compileExecutionV0({
    manifestInference: null,
  });
  assert.equal(compiled, null);
}

{
  const generated = buildExecutionV0FromKernelPath('gemma3-f16-fused-f32a-online');
  assert.ok(generated);
  assert.ok(generated.execution.steps.length > 0);

  const compiled = compileExecutionV0({
    modelId: 'kernel-path-generated',
    numLayers: 4,
    manifestInference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: generated.sessionDefaults,
      execution: generated.execution,
    },
  });

  assert.ok(compiled);
  const prefillAttention = compiled.resolvedSteps.prefill.find(
    (step) => step.op === 'attention' && step.phase === 'prefill'
  );
  assert.ok(prefillAttention);
  assert.equal(prefillAttention.precision.outputDtype, 'f32');
}

{
  const compiled = compileExecutionV0({
    modelId: 'unit-test',
    numLayers: 4,
    manifestInference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: sessionDefaultsFor([
        { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        { kernel: 'matmul_f16.wgsl', entry: 'main' },
      ]),
      execution: {
        steps: [
          {
            id: 'attn_main',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            entry: 'main',
            kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
          },
          {
            id: 'ffn_main',
            phase: 'both',
            section: 'layer',
            op: 'ffn',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'matmul_f16.wgsl',
            entry: 'main',
            kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
          },
        ],
        policies: DEFAULT_POLICIES,
      },
    },
    runtimeInference: {
      session: {
        decodeLoop: {
          batchSize: 12,
        },
      },
    },
  });

  assert.ok(compiled);
  assert.equal(compiled.runtimeInferencePatch.compute.activationDtype, 'f16');
  assert.equal(compiled.runtimeInferencePatch.kvcache.kvDtype, 'f16');
  assert.equal(compiled.runtimeInferencePatch.batching.batchSize, 12);
  assert.equal(compiled.resolvedSteps.prefill.length, 2);
  assert.equal(compiled.resolvedSteps.decode.length, 2);
  assert.equal(compiled.runtimeInferencePatch.kernelPath.id, 'unit-test-execution-v0');
  assert.equal(compiled.runtimeInferencePatch.pipeline.steps.length, 2);
  assert.equal(compiled.resolvedSources.steps.attn_main['precision.inputDtype'].source, 'derived');
  assert.equal(
    compiled.resolvedSources.session['sessionDefaults.compute.defaults.activationDtype'].source,
    'manifest'
  );
}

{
  const compiled = compileExecutionV0({
    modelId: 'hybrid-conv-attn',
    numLayers: 4,
    manifestInference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: sessionDefaultsFor([
        { kernel: 'matmul_f16.wgsl', entry: 'main' },
        { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
      ]),
      execution: {
        steps: [
          {
            id: 'conv_mix',
            phase: 'both',
            section: 'layer',
            op: 'conv',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'matmul_f16.wgsl',
            kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
          },
          {
            id: 'attn_mix',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
          },
          {
            id: 'ffn_mix',
            phase: 'both',
            section: 'layer',
            op: 'ffn',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'matmul_f16.wgsl',
            kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
          },
        ],
        policies: DEFAULT_POLICIES,
      },
    },
  });

  assert.ok(compiled);
  assert.deepEqual(
    compiled.resolvedSteps.prefill.map((step) => step.id),
    ['conv_mix', 'attn_mix', 'ffn_mix']
  );
  assert.ok(compiled.runtimeInferencePatch.pipeline.steps.some((step) => step.op === 'conv'));
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'dtype-mismatch-test',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        ]),
        execution: {
          steps: [
            {
              id: 'attn_requires_f32',
              phase: 'both',
              section: 'layer',
              op: 'attention',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'attention_streaming_f16.wgsl',
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
              precision: {
                inputDtype: 'f32',
                outputDtype: 'f32',
              },
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /Insert explicit cast step/
  );
}

{
  const compiled = compileExecutionV0({
    modelId: 'dtype-cast-test',
    manifestInference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: sessionDefaultsFor([
        { kernel: 'attention_streaming_f16kv.wgsl', entry: 'main' },
      ], 'f32'),
      execution: {
        steps: [
          {
            id: 'cast_up',
            phase: 'both',
            section: 'layer',
            op: 'cast',
            src: 'state',
            dst: 'state',
            layers: 'all',
            toDtype: 'f32',
          },
          {
            id: 'attn_f32',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16kv.wgsl',
            kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
            precision: {
              inputDtype: 'f32',
              outputDtype: 'f32',
            },
          },
          {
            id: 'cast_down',
            phase: 'both',
            section: 'layer',
            op: 'cast',
            src: 'state',
            dst: 'state',
            layers: 'all',
            fromDtype: 'f32',
            toDtype: 'f16',
          },
        ],
        policies: DEFAULT_POLICIES,
      },
    },
  });

  assert.ok(compiled);
  assert.equal(compiled.resolvedSteps.prefill[0].op, 'cast');
  assert.equal(compiled.resolvedSteps.prefill[2].precision.outputDtype, 'f16');
  assert.equal(compiled.runtimeInferencePatch.pipeline.steps.length, 3);
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'explicit-kv-override-mismatch',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16kv.wgsl', entry: 'main' },
        ], 'f32'),
        execution: {
          steps: [
            {
              id: 'attn_prefill',
              phase: 'prefill',
              section: 'layer',
              op: 'attention',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'attention_streaming_f16kv.wgsl',
              entry: 'main',
              kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
      runtimeInference: {
        session: {
          kvcache: {
            kvDtype: 'f32',
          },
        },
      },
    }),
    /Inline kernelPath attention kernel "attention_streaming_f16kv\.wgsl" requires activationDtype="f32" and kvcache\.kvDtype="f16"/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'bdpa-prefill-contract-mismatch',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: {
          ...sessionDefaultsFor([
            { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
          ], 'f16'),
          kvcache: {
            layout: 'bdpa',
            kvDtype: 'f16',
            pageSize: 128,
            windowSize: 1024,
            bdpaVocabSize: 4096,
          },
        },
        execution: {
          steps: [
            {
              id: 'prefill_attn',
              phase: 'prefill',
              section: 'layer',
              op: 'attention',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'attention_streaming_f16.wgsl',
              entry: 'main',
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /sessionDefaults\.kvcache\.layout="bdpa" is decode-only, but step "prefill_attn" declares prefill attention/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'schema-mismatch',
      manifestInference: {
        schema: 'doppler.execution/v1',
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
            },
          ],
          policies: {
            precisionPrecedence: 'step_then_kernel_profile_then_session_default',
            unsupportedPrecision: 'error',
            dtypeTransition: 'require_cast_step',
            unresolvedKernel: 'error',
          },
        },
      },
    }),
    /manifest\.inference\.schema must be "doppler\.execution\/v0"/
  );
}

{
  const compiled = compileExecutionV0({
    modelId: 'patch-order',
    manifestInference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: sessionDefaultsFor([
        { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        { kernel: 'matmul_f16.wgsl', entry: 'main' },
      ]),
      execution: {
        steps: [
          {
            id: 'a',
            phase: 'both',
            section: 'layer',
            op: 'attention',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'attention_streaming_f16.wgsl',
            kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
          },
          {
            id: 'b',
            phase: 'both',
            section: 'layer',
            op: 'ffn',
            src: 'state',
            dst: 'state',
            layers: 'all',
            kernel: 'matmul_f16.wgsl',
            kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
          },
        ],
        policies: DEFAULT_POLICIES,
      },
    },
    runtimeInference: {
      executionPatch: {
        set: [],
        remove: [],
        add: [
          {
            insertAfter: 'a',
            step: {
              id: 'x',
              phase: 'both',
              section: 'layer',
              op: 'noop',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'matmul_f16.wgsl',
              kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
            },
          },
          {
            insertAfter: 'a',
            step: {
              id: 'y',
              phase: 'both',
              section: 'layer',
              op: 'noop',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'matmul_f16.wgsl',
              kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
            },
          },
        ],
      },
    },
  });

  assert.ok(compiled);
  assert.deepEqual(
    compiled.resolvedSteps.prefill.map((step) => step.id),
    ['a', 'x', 'y', 'b']
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'patch-immutable',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        ]),
        execution: {
          steps: [
            {
              id: 'a',
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
          policies: DEFAULT_POLICIES,
        },
      },
      runtimeInference: {
        executionPatch: {
          set: [
            { id: 'a', op: 'ffn' },
          ],
          remove: [],
          add: [],
        },
      },
    }),
    /cannot mutate "op"/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'patch-anchor-after-remove',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
          { kernel: 'matmul_f16.wgsl', entry: 'main' },
        ]),
        execution: {
          steps: [
            {
              id: 'a',
              phase: 'both',
              section: 'layer',
              op: 'attention',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'attention_streaming_f16.wgsl',
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
            },
            {
              id: 'b',
              phase: 'both',
              section: 'layer',
              op: 'ffn',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'matmul_f16.wgsl',
              kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
      runtimeInference: {
        executionPatch: {
          set: [],
          remove: [{ id: 'a' }],
          add: [
            {
              insertAfter: 'a',
              step: {
                id: 'x',
                phase: 'both',
                section: 'layer',
                op: 'noop',
                src: 'state',
                dst: 'state',
                layers: 'all',
                kernel: 'matmul_f16.wgsl',
                kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
              },
            },
          ],
        },
      },
    }),
    /anchor "a" not found/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'duplicate-kernel-profile',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: {
          ...sessionDefaultsFor([
            { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
          ]),
          compute: {
            ...sessionDefaultsFor([
              { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
            ]).compute,
            kernelProfiles: [
              { kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main') },
              { kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main') },
            ],
          },
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
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /duplicate kernel profile/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'missing-kernelref',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        ]),
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
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /requires kernelRef/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'kernelref-binding-mismatch',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
          { kernel: 'matmul_f16.wgsl', entry: 'main' },
        ]),
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
              kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /kernelRef does not match kernel binding/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'missing-src',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        ]),
        execution: {
          steps: [
            {
              id: 'attn',
              phase: 'both',
              section: 'layer',
              op: 'attention',
              dst: 'state',
              layers: 'all',
              kernel: 'attention_streaming_f16.wgsl',
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /\.src must be a non-empty string/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'strict-runtime-overlay',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16.wgsl', entry: 'main' },
        ]),
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
          policies: DEFAULT_POLICIES,
        },
      },
      runtimeInference: {
        compute: { activationDtype: 'f16' },
      },
    }),
    /runtime\.inference overlay supports only session, executionPatch/
  );
}

{
  assert.throws(
    () => compileExecutionV0({
      modelId: 'boundary-mismatch',
      manifestInference: {
        schema: 'doppler.execution/v0',
        sessionDefaults: sessionDefaultsFor([
          { kernel: 'attention_streaming_f16kv.wgsl', entry: 'main' },
          { kernel: 'matmul_f16.wgsl', entry: 'main' },
        ], 'f16'),
        execution: {
          steps: [
            {
              id: 'prefill_attn',
              phase: 'prefill',
              section: 'layer',
              op: 'attention',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'attention_streaming_f16kv.wgsl',
              entry: 'main',
              kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
              precision: {
                outputDtype: 'f32',
              },
            },
            {
              id: 'decode_matmul',
              phase: 'decode',
              section: 'layer',
              op: 'ffn',
              src: 'state',
              dst: 'state',
              layers: 'all',
              kernel: 'matmul_f16.wgsl',
              entry: 'main',
              kernelRef: kernelRef('matmul_f16.wgsl', 'main'),
              precision: {
                inputDtype: 'f16',
              },
            },
          ],
          policies: DEFAULT_POLICIES,
        },
      },
    }),
    /decode step "decode_matmul" reads carried slot "state" as f16 but prefill left f32/
  );
}

{
  const compiled = compileExecutionV0({
    modelId: 'trace-runtime-patch',
    manifestInference: {
      schema: 'doppler.execution/v0',
      sessionDefaults: sessionDefaultsFor([
        { kernel: 'attention_streaming_f16kv.wgsl', entry: 'main' },
      ], 'f32'),
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
            kernel: 'attention_streaming_f16kv.wgsl',
            kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
          },
        ],
        policies: DEFAULT_POLICIES,
      },
    },
    runtimeInference: {
      executionPatch: {
        set: [
          {
            id: 'attn',
            precision: {
              outputDtype: 'f32',
            },
          },
        ],
        remove: [],
        add: [],
      },
    },
  });

  assert.equal(compiled.resolvedSources.steps.attn['precision.outputDtype'].source, 'runtime.patch');
}
