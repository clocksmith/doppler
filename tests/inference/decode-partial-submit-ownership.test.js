import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/generator/decode.js', import.meta.url).href;
const probe = globalThis.partialDecodeOwnershipProbe = {};
const modules = {
  '../../../../gpu/device.js': 'export const getDevice=()=>partialDecodeOwnershipProbe.device; export const setTrackSubmits=()=>{};',
  '../../../../memory/buffer-pool.js': `export const releaseBuffer=b=>partialDecodeOwnershipProbe.release(b);
    export const discardBuffer=releaseBuffer; export const readBuffer=()=>{throw Error('unexpected readback');};`,
  '../../../../gpu/command-recorder.js': `export const createCommandRecorder=()=>partialDecodeOwnershipProbe.recorder;
    export const createProfilingRecorder=createCommandRecorder; export class CommandRecorder {}`,
  '../embed.js': 'export const embed=async()=>({buffer:partialDecodeOwnershipProbe.hidden,dtype:"f32",shape:[1,4]});',
  '../layer.js': `export const hasConvLayers=()=>false;
    export const processLayer=async()=>partialDecodeOwnershipProbe.output;`,
  '../per-layer-inputs.js': 'export const preparePerLayerInputs=async()=>null; export const prefetchPerLayerRow=()=>{};',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) {
    return { url: 'data:text/javascript,' + encodeURIComponent(modules[specifier]), shortCircuit: true };
  }
  return next(specifier, context);
} });

try {
  const { decodeStep } = await import(target);
  for (const borrowed of [false, true]) {
    const hidden = { __dopplerFakeGPUBuffer: true, size: 16 };
    const output = { __dopplerFakeGPUBuffer: true, size: 16 };
    const live = new Set([hidden, output]);
    const tracked = new Set();
    let settle;
    let enteredAbort;
    const submitted = new Promise(resolve => { settle = resolve; });
    const abortEntered = new Promise(resolve => { enteredAbort = resolve; });
    probe.hidden = hidden;
    probe.output = output;
    probe.release = buffer => assert(live.delete(buffer), 'release each owned buffer exactly once');
    probe.device = { queue: { onSubmittedWorkDone: () => submitted } };
    probe.recorder = {
      trackTemporaryBuffer(buffer) { tracked.add(buffer); },
      submitPartial() { throw Error('encoder recreation failed after submitted prefix'); },
      async abort() {
        enteredAbort();
        await submitted;
        for (const buffer of tracked) probe.release(buffer);
        tracked.clear();
      },
      async completeDeferredCleanup() {},
    };
    const state = {
      modelConfig: { hiddenSize: 4, vocabSize: 8, numLayers: 2 },
      weights: new Map([['embed', new Float32Array(32)]]),
      stats: {}, currentSeqLen: 0, decodeStepCount: 0, useGPU: true,
      runtimeConfig: {
        inference: {
          sampling: { greedyThreshold: 0.01 }, compute: { activationDtype: 'f32' },
          session: { usePreparedDecodeWorkspace: false, decodeLoop: { layerSubmitInterval: 1 } },
        },
        shared: { debug: { probes: null } },
      },
      decodeBuffers: {
        resetPingPong() {}, swapPingPong() {},
        getHiddenBuffer: () => borrowed ? hidden : null,
        getOutputHiddenBuffer: () => null,
      },
    };
    const helpers = {
      buildLayerContext: () => ({ decodeBuffers: state.decodeBuffers }),
      releaseSharedAttentionState() {},
    };
    const work = decodeStep(state, [1], {
      executionPlan: { activationDtype: 'f32' }, suppressTokenIds: [],
    }, helpers);
    const rejected = assert.rejects(work, /encoder recreation failed after submitted prefix/);
    await abortEntered;
    assert.deepEqual([...live], [hidden, output], 'submitted prefix retains activations until queue settlement');
    settle();
    await rejected;
    assert.deepEqual([...live], borrowed ? [hidden] : [], 'cleanup releases owned activations and preserves borrowed ones');
  }
} finally {
  hooks.deregister();
  delete globalThis.partialDecodeOwnershipProbe;
}
console.log('decode-partial-submit-ownership: passed (CPU executor substitutions)');
