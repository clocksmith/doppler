import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/index.js';
import { buildLayerContext } from '../../src/inference/pipelines/text/generator-helpers.js';

function createMinimalState() {
  const runtimeConfig = createDopplerConfig().runtime;
  runtimeConfig.shared.debug.pipeline.layers = [1, 3];

  return {
    modelConfig: {
      rmsNormWeightOffset: false,
      perLayerInputsSession: null,
    },
    runtimeConfig,
    debugFlags: {
      loggedStages: new Set(),
    },
    weights: new Map(),
    kvCache: null,
    currentSeqLen: 0,
    currentTokenIds: null,
    useGPU: true,
    debug: true,
    stats: {},
    ropeFreqsCos: null,
    ropeFreqsSin: null,
    ropeLocalCos: null,
    ropeLocalSin: null,
    linearAttentionRuntime: null,
    convLayerStates: new Map(),
    layerPipelinePlan: null,
    expertWeights: new Map(),
    dopplerLoader: null,
    moeRouter: null,
    layerRouterWeights: null,
    decodeBuffers: null,
    lora: null,
    executionV1State: null,
    finitenessBuffer: null,
    decodeStepCount: 0,
    resolvedKernelPath: null,
  };
}

const executionPlan = {
  activationDtype: 'f16',
  kernelPath: null,
  finitenessGuardEnabled: false,
  finitenessAbsThreshold: null,
};

{
  const state = createMinimalState();
  const context = buildLayerContext(state, null, false, undefined, undefined, executionPlan);

  assert.deepEqual(context.debugLayers, [1, 3]);
  assert.deepEqual(context.debugFlags?.debugLayers, [1, 3]);
  assert.notStrictEqual(context.debugFlags, state.debugFlags);
  assert.equal(state.debugFlags.debugLayers, undefined);
  assert.strictEqual(context.debugFlags?.loggedStages, state.debugFlags.loggedStages);
}

{
  const state = createMinimalState();
  const context = buildLayerContext(state, null, false, [2], undefined, executionPlan);

  assert.deepEqual(context.debugLayers, [2]);
  assert.deepEqual(context.debugFlags?.debugLayers, [2]);
}
