import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  buildExecutionContractArtifact,
  extractExecutionContractFacts,
  validateExecutionContractFacts,
  validateManifestExecutionContract,
} from '../../src/config/execution-contract-check.js';
import { buildKernelRefFromKernelEntry } from '../../src/config/kernels/kernel-ref.js';
import { validateManifest } from '../../src/formats/rdrr/validation.js';

const translateGemmaManifest = JSON.parse(
  fs.readFileSync('models/curated/gemma-3-270m-it-wq4k-ef16-hf16/manifest.json', 'utf8')
);

function kernelRef(kernel, entry = 'main') {
  return buildKernelRefFromKernelEntry(kernel, entry);
}

{
  const result = validateManifestExecutionContract(translateGemmaManifest);
  assert.equal(result.ok, true);
  assert.deepEqual(
    result.checks.map((entry) => entry.ok),
    [true, true]
  );
}

{
  const conflictingManifest = structuredClone(translateGemmaManifest);
  conflictingManifest.inference.sessionDefaults.kvcache.layout = 'bdpa';
  conflictingManifest.inference.sessionDefaults.decodeLoop = {
    ...(conflictingManifest.inference.sessionDefaults.decodeLoop ?? {}),
  };
  conflictingManifest.inference.sessionDefaults.decodeLoop.batchSize = 16;
  delete conflictingManifest.inference.sessionDefaults.decodeLoop.disableCommandBatching;

  const facts = extractExecutionContractFacts(conflictingManifest);
  assert.equal(facts.session.layout, 'bdpa');
  assert.equal(facts.session.decodeBatchSize, 16);

  const executionContract = validateExecutionContractFacts(facts);
  assert.equal(executionContract.ok, false);
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('decode-only') && message.includes('prefill attention')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('disableCommandBatching=true')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('batchSize <= 1')
    )
  );
  assert.ok(
    executionContract.errors.some((message) =>
      message.includes('maxSeqLen <= 2048')
    )
  );

  const validation = validateManifest(conflictingManifest);
  assert.equal(validation.valid, false);
  assert.ok(
    validation.errors.some((message) =>
      message.includes('decode-only') && message.includes('prefill attention')
    )
  );
}

{
  const executionV0Manifest = {
    modelId: 'execution-v0-contract-artifact',
    modelType: 'transformer',
    architecture: {
      headDim: 128,
      maxSeqLen: 4096,
      numLayers: 2,
    },
    inference: {
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
            {
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
            },
          ],
        },
        kvcache: {
          layout: 'paged',
          kvDtype: 'f16',
        },
        decodeLoop: {
          batchSize: 4,
          stopCheckMode: 'batch',
          readbackInterval: 1,
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
            entry: 'main',
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
  };

  const artifact = buildExecutionContractArtifact(executionV0Manifest);
  assert.equal(artifact?.ok, true);
  assert.equal(artifact?.executionV0?.kernelProfiles?.ok, true);
  assert.equal(artifact?.executionV0?.graph?.ok, true);
  assert.ok(
    artifact?.checks.some((entry) => entry.id === 'execution-v0-contract-artifact.kernelProfilePinning' && entry.ok)
  );
  assert.ok(
    artifact?.checks.some((entry) => entry.id === 'execution-v0-contract-artifact.slotGraph' && entry.ok)
  );
}

{
  const invalidExecutionV0Manifest = {
    modelId: 'execution-v0-contract-artifact-missing-decode-loop',
    modelType: 'transformer',
    architecture: {
      headDim: 128,
      maxSeqLen: 4096,
      numLayers: 2,
    },
    inference: {
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
            {
              kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
            },
          ],
        },
        kvcache: {
          layout: 'paged',
          kvDtype: 'f16',
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
            entry: 'main',
            kernelRef: kernelRef('attention_streaming_f16.wgsl', 'main'),
          },
        ],
      },
    },
  };

  const artifact = buildExecutionContractArtifact(invalidExecutionV0Manifest);
  assert.equal(artifact?.ok, false);
  assert.ok(
    artifact?.errors.some((message) =>
      message.includes('sessionDefaults.decodeLoop is required')
    )
  );
}

console.log('execution-contract-check.test: ok');
