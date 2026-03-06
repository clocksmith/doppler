import assert from 'node:assert/strict';

import { buildKernelRefFromKernelEntry } from '../../src/config/kernels/kernel-ref.js';
import {
  buildExecutionV0ContractArtifact,
  indexExecutionV0KernelProfiles,
} from '../../src/config/execution-v0-contract-check.js';

function kernelRef(kernel, entry = 'main') {
  return buildKernelRefFromKernelEntry(kernel, entry);
}

{
  const artifact = buildExecutionV0ContractArtifact({
    sessionDefaults: {
      compute: {
        defaults: {
          activationDtype: 'f16',
          mathDtype: 'f32',
          accumDtype: 'f32',
          outputDtype: 'f32',
        },
        kernelProfiles: [
          {
            kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
            precision: {
              inputDtype: 'f32',
              outputDtype: 'f16',
            },
            kvIO: {
              readDtype: 'f16',
              writeDtype: 'f16',
            },
          },
        ],
      },
      kvcache: {
        kvDtype: 'f16',
      },
    },
    execution: {
      steps: [
        {
          id: 'attn',
          op: 'attention',
          kernel: 'attention_streaming_f16kv.wgsl',
          kernelRef: kernelRef('attention_streaming_f16kv.wgsl', 'main'),
          precision: {
            outputDtype: 'f32',
          },
        },
      ],
    },
  }, { modelId: 'execution-v0-contract' });

  assert.equal(artifact.ok, true);
  assert.deepEqual(
    artifact.checks.map((entry) => entry.ok),
    [true, true, true]
  );
  assert.equal(artifact.perStep.attn.precisionSources.inputDtype, 'kernelProfile');
  assert.equal(artifact.perStep.attn.precisionSources.outputDtype, 'manifest');
  assert.equal(artifact.perStep.attn.kvIOSource, 'kernelProfile');
}

{
  assert.throws(
    () => indexExecutionV0KernelProfiles({
      compute: {
        kernelProfiles: [
          { kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main') },
          { kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main') },
        ],
      },
    }),
    /duplicate kernel profile/
  );
}

console.log('execution-v0-contract-check.test: ok');
