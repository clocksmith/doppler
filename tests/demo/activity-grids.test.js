import assert from 'node:assert/strict';
import { createActivationTrace } from '../../src/debug/activation-trace.js';
import { validateRecording } from '../../demo/ui/activation-grid/data.js';
import { encodeRecording } from '../../demo/ui/activation-grid/recording-codec.js';
import { modelPathView } from '../../demo/ui/activation-grid/paths.js';
import { activityView, activityModes, playbackPoints, assertMatchedRuns, attentionRow } from '../../demo/ui/activation-grid/views.js';

const limits = { maxBytes: 4096, maxFrameBytes: 2048, maxFrames: 512, maxGridValues: 64, maxInFlight: 4,
  activityMaxBytes: 8192, activityMaxFrameValues: 64, activityMaxStateValues: 4, activityTokenWindow: 2, activitySummaryBlock: 2,
  weightBlockValues: 2, weightReadValues: 16, weightRowGroupSize: 2, weightRawRows: 2, weightMaxBytes: 1024 };
const trace = createActivationTrace(limits, { layers: 2, width: 3 }, { modelId: 'fixture', manifestHash: 'a'.repeat(64), runtimeVersion: 'fixture', adapters: [] });
const activity = trace.activity;
activity.configure({ intermediateSize: 3, useTiedEmbeddings: true, layerTypes:['linear_attention','full_attention'] });
activity.begin('prefill', 0, [10, 20]);
const down = { id: 'layer_0/ffnDown', layer: 0, role: 'ffnDown', shape: [2, 3], dtype: 'f32', layout: 'row',
  resident: true, byteLength: 24, status: 'captured', reason: null, blockSize: 2,
  rms: [1, 2, 3], raw: [1, 2, 3, -1, 1, 0], rawOffset: 0, columnEnergy: [2, 5, 9], rowGroupSize: 2 };
activity.setWeights([down]);
activity.weightAccess(0, 'ffn_down');
const capture = (role, values, shape, metadata = {}, layer = 0) => activity.capture(role, new Float32Array(values), { layer, dtype: 'f32', shape, metadata });
for (const role of ['ffn-gate', 'ffn-up', 'ffn-product', 'ffn-down-input']) await capture(role, [1, 1, 1, 2, 3, 4], [2, 3]);
await capture('kv-key', [1, 0, 0, 1], [2, 2], {}, 1);
await capture('kv-value', [3, 4, 5, 6], [2, 2], {}, 1);
const attention = { heads: 1, kvHeads: 1, headDim: 2, scale: 1, softcap: 0, slidingWindow: 0, kvLength: 2, causal: true };
await capture('attention-query', [1, 0, 0, 1], [2, 2], attention, 1);
for (const role of ['linear-a', 'linear-b']) await capture(role, [0, 0], [2, 1]);
for (const role of ['linear-a-log', 'linear-dt-bias']) await capture(role, [0], [1, 1]);
await activity.capture('recurrent-memory', new Float32Array([1, 2, 3, 4, 5, 6]), { layer: 0, dtype: 'f32', shape: [2, 3], state: true, metadata: { boundary: 'after-chunk' } });
await capture('convolution-history', [1, 2, 3, 4], [2, 2]);
await activity.logits(new Float32Array([0, Math.log(2), Math.log(3)]));
for (let layer = 0; layer < 2; layer++) trace.capture(new Float32Array([1, 2, 3, 3, 5, 7]), { layer, phase: 'prefill', tokenStart: 0, tokenIds: [10, 20], dtype: 'f32' });
await trace.drain();
const recording = { schema: 'doppler.activation-recording/v1', demoBuild: 'fixture',
  display: { layout: 'layer-major-rgb/v1', palette: 'signed-tanh-rgb/v1', scale: 1 }, trace: trace.snapshot('complete'), inputTokenIds: [10, 20], generatedTokenIds: [] };
validateRecording(recording);
const path = modelPathView(recording, Infinity, 1, 0);
assert.deepEqual(path.outputs, [3,5,7,3,5,7], 'The path heatmap retains layer-major captured values');
assert.equal(path.width, 3);
assert.equal(path.kind, 'linear_attention');
assert(Math.abs(path.nodes.find(n => n.id === 'gate').magnitude - Math.sqrt(29/3)) < 1e-12);
assert(Math.abs(path.nodes.find(n => n.id === 'mixer').magnitude - Math.sqrt(91/6)) < 1e-12,
  'State summary RMS weights partial blocks by their scalar counts');
assert(modelPathView(recording, -1, 1, 0).layers.every(l => l.magnitude === null), 'Uncaptured layers are missing, not invalid');
const attentionPath = modelPathView(recording, Infinity, 1, 1);
assert.equal(attentionPath.kind, 'full_attention');
assert.equal(attentionPath.nodes.find(n => n.id === 'mixer').magnitude, null,
  'Uncaptured attention results remain empty');
assert.equal(modelPathView(recording, -1, 1, 0).nodes.filter(n => n.magnitude !== null).length, 0,
  'A path never shows future captures');
assert(playbackPoints(recording, 'captures', 'recurrent-memory').every(p => p.layer === 0),
  'Memory playback uses state captures, not unrelated operator ticks');
const selection = { atMs: Infinity, position: 1, layer: -1, weightId: '', detail: 'overview', page: 0 };
const view = (mode, changes = {}) => activityView(recording, { ...selection, mode, ...changes });
assert.deepEqual(view('difference').values, [2, 3, 4, 2, 3, 4]);
assert.deepEqual(view('ffn-product').values.slice(0, 3), [2, 3, 4]);
assert(view('ffn-product').values.slice(3).every(v => v === undefined), 'missing layer is not zero');
assert.deepEqual(view('residency').values, [1], 'access is tied to an observed projection dispatch');
assert.equal(view('contributions').underlyingValues, 6);
assert.deepEqual(view('kv-new').values, [0, 1, 5, 6]);
assert.deepEqual(view('kv-cache').values, [1, 0, 3, 4, 0, 1, 5, 6], 'append-only writes retain earlier entries');
const query = recording.trace.activity.frames.find(f => f.role === 'attention-query');
assert.deepEqual(attentionRow(recording.trace.activity.frames, query, 0, 0), [1, 0], 'future key is causally masked');
const row = attentionRow(recording.trace.activity.frames, query, 1, 0);
assert(Math.abs(row[1] - Math.E / (1 + Math.E)) < 1e-12);
assert.equal(view('attention-prefill').values.length, 4);
assert.deepEqual(view('linear-gates').values, [0.5, 0.5]);
assert.deepEqual(view('recurrent-memory').values, [Math.sqrt(2.5), Math.sqrt(12.5), Math.sqrt(30.5)], 'overview covers the full state without retaining its full payload');
assert.deepEqual(view('recurrent-memory', {detail: 'raw'}).values, [1, 2, 3, 4], 'raw window stays bounded and labeled');
assert.deepEqual(view('contributions', { detail: 'raw' }).values, [2, 6, 12, -2, 3, 0]);
const shifted = structuredClone(recording);
shifted.trace.activity.weights[0].raw = [3, -1];
shifted.trace.activity.weights[0].rawOffset = 2;
assert.deepEqual(activityView(shifted, { ...selection, mode: 'contributions', detail: 'raw' }).values,
  [12, -2], 'A saved connection window uses its actual input-column offset');
assert.equal(view('contributions', { weightId: 'not-captured' }).values.length, 0,
  'Selecting an uncaptured tensor must not show another tensor');
assert.equal(view('weights', { layer: 1 }).values.length, 0,
  'The selected layer also limits weight views');
const missingWeights = structuredClone(recording);
missingWeights.trace.activity.weights.push({ ...down, id: 'missing', shape: [4, 3], rms: [], raw: [], columnEnergy: [], status: 'unavailable' });
const atlas = activityView(missingWeights, { ...selection, mode: 'weights' });
assert.equal(atlas.values.length, 9, 'Known missing tensor occupies all six blocks, not one');
assert(atlas.values.slice(3).every(v => v === undefined));
assert.equal(atlas.segments[1].start, 3); assert.equal(atlas.segments[1].end, 9);
assert.equal(view('quantization-error', { weightId: 'not-captured' }).values.length, 0);
const partialState = structuredClone(recording);
const stateFrame = partialState.trace.activity.frames.find(f => f.role === 'recurrent-memory');
stateFrame.summary.values = stateFrame.summary.values.slice(0,1); stateFrame.summary.observedValues = 2;
const stateView = activityView(partialState, { ...selection, mode:'recurrent-memory' });
assert.equal(stateView.values.length, 3); assert.equal(stateView.values[1], undefined,
  'State summaries reserve unobserved blocks instead of implying full coverage');
const energy = view('contributions').values;
assert.deepEqual(energy, [2, Math.sqrt(22.5), Math.sqrt(72)], 'block energy matches an independent explicit product calculation');
const probabilities = view('probabilities').values;
assert(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
assert(Math.abs(probabilities[2] - 0.5) < 1e-7);
assert.equal(activityModes(recording).find(m => m.id === 'expert-routing').reason !== null, true);
assert.equal(activityModes(recording).find(m => m.id === 'quantization-error').reason !== null, true);
assert.deepEqual(playbackPoints(recording, 'tokens').map(p => p.position), [0, 1], 'prefill positions are individually selectable, not new execution events');
assertMatchedRuns(recording, structuredClone(recording));
const relocated = structuredClone(recording); relocated.trace.identity.modelId = 'another-source-url';
assertMatchedRuns(recording, relocated); // The same manifest digest preserves artifact identity across source locations.
assert.equal(view('recurrent-memory', {position: 7}).values.length, 0, 'uncaptured positions never borrow older state');
const mismatched = structuredClone(recording); mismatched.inputTokenIds[1] = 99;
assert.throws(() => assertMatchedRuns(recording, mismatched), /prefixes differ/);
const differentArtifact = structuredClone(recording); differentArtifact.trace.identity.manifestHash = 'b'.repeat(64);
assert.throws(() => assertMatchedRuns(recording, differentArtifact), /exact artifact/);
assert.deepEqual(validateRecording(JSON.parse(JSON.stringify(encodeRecording(recording)))), recording, 'packed recordings retain numerical values and provenance');
const malicious = structuredClone(recording); malicious.trace.activity.frames[0].shape = [2 ** 30];
assert.throws(() => validateRecording(malicious), /Invalid/);
const packed = encodeRecording(recording); packed.trace.activity.frames[0].values.data = '';
assert.throws(() => validateRecording(packed), /packed/);
const brokenAttention = structuredClone(recording); brokenAttention.trace.activity.frames = brokenAttention.trace.activity.frames.filter(f => f.role !== 'kv-key');
const missingRow = attentionRow(brokenAttention.trace.activity.frames, query, 1, 0);
assert(missingRow.every(v => v === undefined), 'missing key never becomes a normalized partial attention row');
const finalPoint = playbackPoints(recording, 'tokens').at(-1);
assert(finalPoint.atMs >= recording.trace.activity.frames.find(f => f.role === 'logits').atMs, 'token playback includes the prediction after the final layer');
console.log('Activity views: raw/aggregate products, causal attention, cache history, state gaps, gates, differences, matching, packed replay and malformed imports passed.');
