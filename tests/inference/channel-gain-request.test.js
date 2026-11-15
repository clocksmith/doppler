import assert from 'node:assert/strict';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';
import { createChannelGainRequest, assertChannelGainIdle, withChannelGainOperation } from '../../src/inference/pipelines/text/channel-gain.js';
import { validateChannelTensor, deriveChannelGains, channelEnergy } from '../../src/gpu/kernels/channel-gain.js';

const tensor = { dtype: 'f32', shape: [2], buffer: { size: 8 } };
const spec = { mode: 'capture', requestId: 'capture-a', gainHash: '', cacheArm: 'base-cache',
  tokenCount: 4, span: { start: 1, stop: 3, templatePositions: [0, 3] },
  layers: [{ layerIdx: 2, channels: 2 }] };
const context = { config: { useMoE: false, gatedActivation: true, hiddenActivation: 'silu' },
  activationDtype: 'f32', phase: 'prefill', currentSeqLen: 0 };
const capture = createChannelGainRequest(spec);
assert.doesNotThrow(() => capture.validateLayer(2, context, 4, 2));
assert.throws(() => capture.validateLayer(2, { ...context, currentSeqLen: 1 }, 4, 2), /fresh full prefill/);
assert.throws(() => capture.validateLayer(2, context, 2, 2), /fresh full prefill/);
assert.throws(() => capture.validateLayer(2, { ...context, phase: 'decode' }, 1, 2), /frozen gains/);
assert.throws(() => capture.validateLayer(2, { ...context, activationDtype: 'f16' }, 4, 2), /ordinary f32/);
assert.throws(() => capture.validateLayer(2, context, 4, 3), /ordinary f32/);
spec.layers[0].channels = 99;
assert.doesNotThrow(() => capture.validateLayer(2, context, 4, 2));
assert.throws(() => capture.takeEnergies(), /incomplete/);
capture.close(); capture.close();
assert.throws(() => capture.has(2), /closed/);
assert.throws(() => createChannelGainRequest({ ...spec, span: { start: 0, stop: 3, templatePositions: [0] } }), /template/);
assert.throws(() => createChannelGainRequest({ ...spec, mode: 'apply', gainHash: 'a'.repeat(64), frozenGains: { gainHash: 'a'.repeat(64) } }), /opaque/);
assert.throws(() => createChannelGainRequest({ ...spec, layers: [{ layerIdx: 2, channels: 2 }, { layerIdx: 2, channels: 2 }] }), /unique/);
assert.throws(() => validateChannelTensor(tensor, 2, 2), /sufficient size/);
await assert.rejects(channelEnergy(null, tensor, { tokens: 1, channels: 2, start: 0, stop: 2 }), /exact nonempty/);
await assert.rejects(deriveChannelGains(null, tensor, tensor, { channels: 2, beta: 1, clip: 1, epsilon: 0 }), /positive epsilon/);
let resets = 0;
let closed = 0;
const state = { isGenerating: false, channelGainRequest: { close() { closed += 1; } } };
PipelineGenerator.prototype.resetGenerationState.call({ _state: state,
  _resetReplayPrefillRuntimeState() { resets += 1; }, _resetDecodeRuntimeState() { resets += 1; } });
assert.equal(closed, 1); assert.equal(resets, 2); assert.equal(state.channelGainRequest, null);
const operationState = {};
let complete;
const operation = withChannelGainOperation(operationState, () => new Promise(resolve => { complete = resolve; }));
assert.throws(() => assertChannelGainIdle(operationState), /active operation/);
await assert.rejects(withChannelGainOperation(operationState, async () => {}), /serialized/);
complete(); await operation;
assert.doesNotThrow(() => assertChannelGainIdle(operationState));
assert.throws(() => assertChannelGainIdle({ isGenerating: true }), /active operation/);
console.log('channel-gain request CPU contracts passed; no GPU dispatch performed');

// Same-pass capture is opt-in and has its own exact, template-free content span.
const pairSpec = { mode: 'capture', requestId: 'pair', gainHash: '', cacheArm: 'base-cache',
  tokenCount: 6, span: { start: 2, stop: 4, templatePositions: [0, 5] },
  captureContentSpan: { start: 1, stop: 5 }, layers: [{ layerIdx: 2, channels: 2 }] };
const pair = createChannelGainRequest(pairSpec);
pairSpec.captureContentSpan.start = 0;
assert.throws(() => pair.takeReferenceEnergies(), /incomplete/);
assert.doesNotThrow(() => pair.validateLayer(2, context, 6, 2));
pair.close(); assert.throws(() => pair.takeReferenceEnergies(), /closed/);
for (const content of [{start:0,stop:5},{start:2,stop:4},{start:3,stop:5},{start:1,stop:7}]) {
  assert.throws(() => createChannelGainRequest({...pairSpec,captureContentSpan:content}), /Same-pass capture/);
}
const plain = createChannelGainRequest({...pairSpec,captureContentSpan:undefined});
assert.throws(() => plain.takeReferenceEnergies(), /absent/);plain.close();
console.log('same-pass capture span and reference-ownership contracts passed');

const rotationSpec = { ...pairSpec, captureContentSpan: { start: 1, stop: 5 }, transformKind: 'rotation', rotationEpsilon: 1e-6, retainFinalLayerReplay: true };
const rotationCapture = createChannelGainRequest(rotationSpec);
assert.throws(() => rotationCapture.captureFinalLayer(2, tensor, null, 6, context, undefined, false), /Unsupported/);
await assert.rejects(rotationCapture.replayFinalLayer(rotationCapture), /settled capture/);
rotationCapture.close();
assert.throws(() => createChannelGainRequest({ ...rotationSpec, layers: [{layerIdx: 1, channels: 2}, {layerIdx: 2, channels: 2}] }), /exactly one/);
console.log('final-layer replay capture preconditions reject unsupported paths and multiple layers');
