import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createResolvedRuntimeSession,
  resolveAttentionRuntimeSession,
} from '../../src/inference/pipelines/text/resolved-runtime-session.js';
import { createObservationContext } from '../../src/inference/observation-context.js';

function createInputs() {
  const kernelPath = {
    id: 'unit-kernel-path',
    decode: { steps: [{ op: 'attention', kernel: 'attention_f32' }] },
    prefill: { steps: [{ op: 'attention', kernel: 'attention_f32' }] },
  };
  const primaryPlan = {
    id: 'primary',
    source: 'configured',
    activationDtype: 'f32',
    kernelPathId: kernelPath.id,
    kernelPathSource: 'execution-v1',
    kernelPath,
    finitenessGuardEnabled: false,
    finitenessOnTrigger: 'error',
    defaultDisableCommandBatching: false,
    defaultDisableMultiTokenDecode: false,
    defaultBatchSize: 1,
    defaultStopCheckMode: 'per-token',
    readbackInterval: 1,
    readbackMode: 'sequential',
    maxBatchDecodeTokens: null,
  };
  return {
    manifest: {
      modelId: 'unit-model',
      modelType: 'transformer',
      architecture: {
        type: 'unit',
      },
      inference: {
        session: {
          compute: {
            defaults: {
              activationDtype: 'f32',
              outputDtype: 'f32',
              mathDtype: 'f32',
              accumDtype: 'f32',
            },
          },
        },
      },
    },
    modelConfig: {
      numLayers: 2,
      hiddenSize: 16,
      numHeads: 2,
      numKVHeads: 1,
      headDim: 8,
    },
    runtimeConfig: {
      inference: {
        compute: {
          activationDtype: 'f32',
        },
        session: {
          useFusedRmsnormWideTile: true,
          kvcache: { kvDtype: 'f32' },
          compute: {
            defaults: {
              activationDtype: 'f32',
              outputDtype: 'f32',
              mathDtype: 'f32',
              accumDtype: 'f32',
            },
          },
        },
      },
    },
    resolvedKernelPath: kernelPath,
    kernelPathSource: 'execution-v1',
    executionV1State: {
      resolvedSteps: {
        prefill: [{ op: 'attention', kernel: 'attention_f32' }],
        decode: [{ op: 'attention', kernel: 'attention_f32' }],
        all: [{ op: 'attention', kernel: 'attention_f32' }],
      },
      laneIntegrity: {
        status: 'matches',
        policy: { name: 'default-noop' },
      },
      appliedTransforms: [],
    },
    executionPlanState: {
      primaryPlan,
      fallbackPlan: null,
      activePlanId: 'primary',
    },
  };
}

test('resolved runtime session is immutable and detached from mutable inputs', () => {
  const inputs = createInputs();
  const resolved = createResolvedRuntimeSession(inputs);
  const originalId = resolved.id;

  inputs.runtimeConfig.inference.session.useFusedRmsnormWideTile = false;
  inputs.resolvedKernelPath.decode.steps[0].kernel = 'attention_changed';

  assert.equal(resolved.runtime.session.useFusedRmsnormWideTile, true);
  assert.equal(resolved.kernelPath.definition.decode.steps[0].kernel, 'attention_f32');
  assert.equal(resolved.id, originalId);
  assert.equal(Object.isFrozen(resolved), true);
  assert.equal(Object.isFrozen(resolved.runtime.session), true);
  assert.throws(() => {
    resolved.runtime.session.useFusedRmsnormWideTile = false;
  }, TypeError);
});

test('attention accepts only the resolved session boundary', () => {
  assert.throws(
    () => resolveAttentionRuntimeSession({}),
    /attention requires a resolved runtime session/
  );
  const resolved = createResolvedRuntimeSession(createInputs());
  assert.equal(
    resolveAttentionRuntimeSession({ resolvedRuntimeSession: resolved }),
    resolved.runtime.session
  );
});

test('observation context keeps command intent out of numeric runtime policy', () => {
  const commandContext = Object.freeze({
    schemaVersion: 1,
    command: 'verify',
    workload: 'inference',
    intent: 'verify',
  });
  const observation = createObservationContext({
    commandContext,
    runtimeConfig: {
      shared: {
        tooling: {
          diagnostics: 'on_failure',
          refactorReceipts: 'always',
        },
        debug: {
          probes: [],
          pipeline: { enabled: false, layers: null },
          kernelTrace: { enabled: false },
        },
      },
    },
  });

  assert.equal(observation.commandContext, commandContext);
  assert.equal(observation.receiptPolicy, 'always');
  assert.equal('inference' in observation, false);
  assert.equal(Object.isFrozen(observation), true);
});
