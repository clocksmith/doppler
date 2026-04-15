import assert from 'node:assert/strict';

import { createDopplerConfig } from '../../src/config/schema/index.js';
import {
  buildLayerContext,
  resolvePerLayerInputsSession,
} from '../../src/inference/pipelines/text/generator-helpers.js';

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

const basePerLayerInputsSession = {
  materialization: 'range_backed',
  rowCache: {
    mode: 'prepared_tokens',
    maxRows: 8,
    maxBytes: 1000,
    decodedDtype: 'f16',
    outputDtype: 'f16',
  },
  prefetch: {
    mode: 'none',
    rowsAhead: 1,
  },
  gpuUpload: {
    mode: 'immediate',
    stagingRows: 1,
  },
  hotCache: {
    mode: 'prepared_tokens',
    maxTokens: 16,
    maxBytes: 1024,
    outputDtype: 'f16',
  },
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

{
  const state = createMinimalState();
  state.modelConfig.perLayerInputsSession = basePerLayerInputsSession;
  state.runtimeConfig.inference.session = state.runtimeConfig.inference.session ?? {};
  state.runtimeConfig.inference.session.perLayerInputs = {
    materialization: 'gpu_split_tables',
  };
  const context = buildLayerContext(state, null, false, undefined, undefined, executionPlan);

  assert.equal(context.perLayerInputsSession?.materialization, 'gpu_split_tables');
  assert.equal(context.perLayerInputsSession?.rowCache?.mode, 'prepared_tokens');
  assert.equal(context.perLayerInputsSession?.prefetch?.mode, 'none');
}

{
  const manifestSession = basePerLayerInputsSession;
  const runtimeSession = {
    materialization: 'gpu_split_tables',
  };
  const resolved = resolvePerLayerInputsSession(manifestSession, runtimeSession);

  assert.equal(resolved?.materialization, 'gpu_split_tables');
  assert.equal(resolved?.rowCache?.maxRows, 8);
  assert.equal(resolved?.hotCache?.maxTokens, 16);
}
