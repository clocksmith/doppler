import assert from 'node:assert/strict';

import { buildKernelRefFromKernelEntry } from '../../src/config/kernels/kernel-ref.js';
import { buildExecutionV0GraphContractArtifact } from '../../src/config/execution-v0-graph-contract-check.js';

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
  const artifact = buildExecutionV0GraphContractArtifact({
    modelId: 'graph-pass',
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
        policies: {
          precisionPrecedence: 'step_then_kernel_profile_then_session_default',
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
      },
    },
  });

  assert.equal(artifact.ok, true);
  assert.deepEqual(
    artifact.checks.map((entry) => entry.ok),
    [true, true]
  );
}

{
  const artifact = buildExecutionV0GraphContractArtifact({
    modelId: 'graph-missing-slot',
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
            src: 'tmp',
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
    },
  });

  assert.equal(artifact.ok, false);
  assert.deepEqual(
    artifact.checks.map((entry) => entry.ok),
    [false, true]
  );
}

{
  const artifact = buildExecutionV0GraphContractArtifact({
    modelId: 'graph-boundary-mismatch',
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
        policies: {
          precisionPrecedence: 'step_then_kernel_profile_then_session_default',
          unsupportedPrecision: 'error',
          dtypeTransition: 'require_cast_step',
          unresolvedKernel: 'error',
        },
      },
    },
  });

  assert.equal(artifact.ok, false);
  assert.deepEqual(
    artifact.checks.map((entry) => entry.ok),
    [true, false]
  );
}

console.log('execution-v0-graph-contract-check.test: ok');
