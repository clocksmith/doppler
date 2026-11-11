import assert from 'node:assert/strict';
import { createActivationTrace } from '../../src/debug/activation-trace.js';
import { createActivityTrace, resolveActivityCapture } from '../../src/debug/activity-trace.js';
import { validateRecording, colorGrid, heatmapColor, heatmapColors, rankProbabilities } from '../../demo/ui/activation-grid/data.js';
import { encodeRecording } from '../../demo/ui/activation-grid/recording-codec.js';
import { modelPathView } from '../../demo/ui/activation-grid/paths.js';
import { activityView, activityModes, playbackPoints, assertMatchedRuns, attentionRow, attentionGridView, viewPositions, missingReason, omissionSummary, viewCaptureRoles } from '../../demo/ui/activation-grid/views.js';

const limits = { maxBytes: 4096, maxFrameBytes: 2048, maxFrames: 512, maxGridValues: 64, maxInFlight: 4,
  activityMaxBytes: 8192, activityMaxFrameValues: 64, activityMaxStateValues: 4, activityTokenWindow: 2, activitySummaryBlock: 2,
  activitySelection: { roles: null, layers: null, tokens: null }, activityRoleMaxBytes: {},
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
// A view's timeline holds only positions it can draw, so playback never lands on an empty grid.
assert.equal(viewPositions(recording, 'difference'), null, 'Layer-output views can draw every recorded position');
assert.deepEqual([...viewPositions(recording, 'probabilities')], [1], 'A prediction belongs to the one position it was made at');
assert.deepEqual(playbackPoints(recording, 'tokens', 'probabilities').map(p => p.position), [1]);
assert.deepEqual(playbackPoints(recording, 'tokens', 'kv-cache').map(p => p.position), [0, 1]);
assert.deepEqual([...viewPositions(recording, 'linear-gates')], [0, 1], 'Fixed gate parameters do not limit the positions');
{
  const withoutValues = structuredClone(recording);
  for (const frame of withoutValues.trace.activity.frames) if (frame.role === 'logits') frame.values = null;
  assert.match(activityModes(withoutValues).find(m => m.id === 'probabilities').reason, /no capture/,
    'A view with nothing to draw is reported unavailable rather than offered empty');
}
// Missing values are explained from the recording itself, with the nearest token that has them.
assert.deepEqual(missingReason(recording, 'kv-new', 0), { kind: 'none', nearest: 0, text: '' });
assert.deepEqual(missingReason(recording, 'probabilities', 0), { kind: 'elsewhere', nearest: 1, text: 'Saved only for tokens 1–1.' });
{
  const overBudget = structuredClone(recording);
  for (const frame of overBudget.trace.activity.frames) {
    if (frame.role === 'kv-key') { frame.status = 'dropped'; frame.values = null; frame.reason = 'Activity recording budget reached; inference continues.'; }
  }
  const dropped = missingReason(overBudget, 'kv-new', 1);
  assert.equal(dropped.kind, 'dropped');
  assert.match(dropped.text, /reached its size limit \(0 MB\) at token 0, so this view was not saved after that/);
  const failed = structuredClone(recording);
  for (const frame of failed.trace.activity.frames) if (frame.role === 'kv-value') { frame.status = 'failed'; frame.values = null; frame.reason = 'readback lost'; }
  assert.equal(missingReason(failed, 'kv-new', 1).text, 'Saving this view failed at token 1: readback lost');
  const windowed = structuredClone(recording);
  for (const frame of windowed.trace.activity.frames) {
    if (frame.role === 'ffn-gate') { frame.tokenStart = 1; frame.tokenIds = [20]; frame.fullShape = [1, 3]; frame.values = frame.values.slice(0, 3); frame.metadata = { ...frame.metadata, originalShape: [2, 3] }; }
  }
  assert.deepEqual(missingReason(windowed, 'ffn-gate', 0), { kind: 'window', nearest: 1, text: 'Saved for the last 1 prompt tokens only (tokens 1–1).' });
}
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
const attentionGrid = attentionGridView(recording, Infinity, 1);
assert.deepEqual(attentionGrid.rows, [{ layer: 1, head: 0 }]);
assert.equal(attentionGrid.width, 2);
assert.deepEqual(attentionGrid.values, row, 'The default attention grid preserves head/key coordinates');
assert.deepEqual(attentionGridView(recording, Infinity, 0).values, [1, 0], 'The grid follows the shared query position and causal mask');
assert.equal(attentionGridView(recording, -1, 1).values.length, 0, 'Future captures never leak into the grid');
assert.equal(attentionGridView(recording, Infinity, 1, 0).values.length, 0, 'A linear layer is not fabricated into attention');
{
  const missingKey = structuredClone(recording);
  missingKey.trace.activity.frames = missingKey.trace.activity.frames.filter(f => f.role !== 'kv-key');
  assert(attentionGridView(missingKey, Infinity, 1).values.every(v => v === undefined), 'Missing keys never produce normalized partial attention');
}
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
assert.deepEqual(view('probabilities').distribution.entries.map(e => e.tokenId), [2, 1, 0]);
{
  const sparse = new Array(248320).fill(0);
  sparse[12515] = 0.79; sparse[200000] = 0.2;
  for (let i = 0; i < 100; i++) sparse[i] = 0.0001;
  const ranked = rankProbabilities(sparse);
  assert.deepEqual(ranked.entries.slice(0, 2), [{ tokenId: 12515, probability: 0.79 }, { tokenId: 200000, probability: 0.2 }], 'Sparse probabilities are surfaced without losing their token IDs');
  assert(Math.abs(ranked.entries.reduce((sum, e) => sum + e.probability, ranked.remaining) - 1) < 1e-12, 'The omitted tail remains part of the whole-vocabulary distribution');
  assert.equal(rankProbabilities([0.5, undefined]), null, 'Incomplete distributions are never renormalized');
  assert.equal(rankProbabilities([]), null);
}
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

// Selective capture: a recording can keep chosen kinds, layers and tokens, with a
// budget per kind, and says what it left out instead of leaving unexplained gaps.
{
  const rows = (/** @type {number} */ count, width = 2) => Array.from({ length: count * width }, (_, i) => i);
  const record = (recorder, role, values, shape, layer = 0, options = {}) => recorder.capture(role, new Float32Array(values), { layer, dtype: 'f32', shape, ...options });
  const all = { roles: null, layers: null, tokens: null };

  assert.throws(() => createActivityTrace({ ...limits, activitySelection: undefined }, 0), /activitySelection/, 'a recorder never assumes what to capture');
  assert.throws(() => createActivityTrace({ ...limits, activityRoleMaxBytes: { 'not-a-kind': 8 } }, 0), /capture kind/);

  const chosen = createActivityTrace({ ...limits, activitySelection: { roles: ['ffn-gate', 'kv-key', 'linear-a-log'], layers: [1], tokens: { from: 2, to: 4 } } }, performance.now());
  chosen.begin('prefill', 0, [10, 20, 30, 40, 50]);
  await record(chosen, 'ffn-gate', rows(5), [5, 2], 1);
  await record(chosen, 'ffn-gate', rows(5), [5, 2], 0);
  await record(chosen, 'ffn-up', rows(5), [5, 2], 1);
  await record(chosen, 'kv-key', rows(5), [5, 2], 1);
  await record(chosen, 'linear-a-log', [7], [1, 1], 1);
  chosen.begin('decode', 5, [60]);
  await record(chosen, 'ffn-gate', rows(1), [1, 2], 1);
  await record(chosen, 'linear-a-log', [7], [1, 1], 1);
  const kept = chosen.snapshot();
  const gate = kept.frames.find(f => f.role === 'ffn-gate');
  assert.deepEqual([gate.tokenStart, gate.tokenIds, gate.fullShape, gate.values], [2, [30, 40], [2, 2], [4, 5, 6, 7]], 'a token range keeps exactly its rows');
  const key = kept.frames.find(f => f.role === 'kv-key');
  assert.deepEqual([key.tokenStart, key.fullShape, key.values.length], [0, [4, 2], 8], 'keys keep every earlier token, because later tokens attend to them');
  assert.deepEqual(kept.frames.map(f => f.role), ['ffn-gate', 'kv-key', 'linear-a-log'], 'what is left out takes no frame');
  assert.equal(kept.frames[2].metadata.constant, true);
  assert.deepEqual(kept.coverage['ffn-gate'].notSelected, { role: 0, layer: 1, token: 1 });
  assert.deepEqual([kept.coverage['ffn-gate'].captured, kept.coverage['ffn-up'].notSelected.role], [1, 1]);
  assert.deepEqual([kept.coverage['linear-a-log'].captured, kept.coverage['linear-a-log'].reused], [1, 1], 'a fixed per-layer value is recorded once, outside any token range');
  assert.equal(kept.recordedBytes, (4 + 8 + 1) * 8);
  assert.deepEqual(kept.selection, { roles: ['ffn-gate', 'kv-key', 'linear-a-log'], layers: [1], tokens: { from: 2, to: 4 } });
  assert.equal(omissionSummary({ trace: { activity: kept } }),
    'left out by choice: ffn-up · 1 captures on layers not selected · 1 captures outside the selected tokens · 1 repeats of fixed per-layer values recorded once');

  const capped = createActivityTrace({ ...limits, activitySelection: all, activityRoleMaxBytes: { 'ffn-gate': 32 } }, performance.now());
  for (const [position, token] of [10, 20, 30].entries()) {
    capped.begin('decode', position, [token]);
    await record(capped, 'ffn-gate', rows(1), [1, 2]);
    await record(capped, 'ffn-up', rows(1), [1, 2]);
  }
  const overRole = capped.snapshot();
  const lost = overRole.frames.filter(f => f.status === 'dropped');
  assert.deepEqual(lost.map(f => [f.role, f.tokenStart, f.metadata.limit]), [['ffn-gate', 2, 'activityRoleMaxBytes']], 'one kind running out leaves the others recording');
  assert.match(lost[0].reason, /budget for ffn-gate reached/);
  assert.deepEqual([overRole.coverage['ffn-gate'].dropped, overRole.coverage['ffn-gate'].firstDroppedToken, overRole.coverage['ffn-up'].captured], [1, 2, 3]);
  assert.equal(omissionSummary({ trace: { activity: overRole } }), 'out of budget: ffn-gate from token 2');

  // A request narrows its policy and never widens it.
  const policy = { ...limits, activitySelection: { roles: ['ffn-gate', 'logits'], layers: null, tokens: { from: 0, to: 8 } }, activityRoleMaxBytes: { logits: 64 } };
  assert.equal(resolveActivityCapture(policy, undefined), policy);
  const narrowed = resolveActivityCapture(policy, { roles: ['logits'], layers: [0], tokens: { from: 2, to: 6 }, maxBytes: 1024, roleMaxBytes: { logits: 32 } });
  assert.deepEqual([narrowed.activitySelection, narrowed.activityMaxBytes, narrowed.activityRoleMaxBytes],
    [{ roles: ['logits'], layers: [0], tokens: { from: 2, to: 6 } }, 1024, { logits: 32 }]);
  assert.equal(policy.activityMaxBytes, 8192, 'the policy itself is left untouched');
  assert.throws(() => resolveActivityCapture(policy, { roles: ['kv-key'] }), /does not record the kind "kv-key"/);
  assert.throws(() => resolveActivityCapture(policy, { tokens: { from: 2, to: null } }), /outside the range/);
  assert.throws(() => resolveActivityCapture(policy, { maxBytes: 8193 }), /between 1 and the policy's 8192 bytes/);
  assert.throws(() => resolveActivityCapture(policy, { roleMaxBytes: { logits: 65 } }), /exceeds the 64 bytes/);
  assert.throws(() => resolveActivityCapture(policy, { layers: [-1] }), /distinct valid entries/);
  assert.deepEqual(resolveActivityCapture(policy, { roles: [] }).activitySelection.roles, [], 'an empty list records none, unlike null');

  // Recording one view keeps exactly what that view draws from.
  assert.deepEqual(viewCaptureRoles('linear-gates'), ['linear-a', 'linear-b', 'linear-a-log', 'linear-dt-bias']);
  assert.deepEqual(viewCaptureRoles('difference'), [], 'layer outputs are always recorded, so this view needs no activity kinds');
  assert.deepEqual(viewCaptureRoles('residency'), ['weight-access']);
  assert.equal(viewCaptureRoles('model-paths'), null, 'the model map draws from every kind');
  const none = createActivityTrace({ ...limits, activitySelection: { ...all, roles: [] } }, performance.now());
  none.begin('decode', 0, [10]);
  await record(none, 'ffn-gate', rows(1), [1, 2]);
  assert.deepEqual([none.snapshot().frames.length, none.snapshot().coverage['ffn-gate'].notSelected.role], [0, 1]);
  assert.throws(() => resolveActivityCapture(policy, { budget: 1 }), /Unknown capture selection field "budget"/);
  assert.throws(() => resolveActivityCapture({ maxBytes: 1 }, { roles: ['logits'] }), /records model activity/);

  // The viewer tells "left out by choice" apart from "ran out of budget".
  assert.deepEqual(recording.trace.activity.selection, all);
  const without = structuredClone(recording);
  without.trace.activity.selection = { ...all, roles: ['kv-key', 'kv-value', 'attention-query'] };
  without.trace.activity.frames = without.trace.activity.frames.filter(f => f.role !== 'logits');
  assert.equal(missingReason(without, 'probabilities', 1).kind, 'not-selected');
  assert.match(activityModes(without).find(m => m.id === 'probabilities').reason, /set to leave this view out/);
  assert.equal(missingReason(without, 'kv-new', 1).kind, 'none', 'a selected view is unaffected');

  const ranged = structuredClone(recording);
  ranged.trace.activity.selection = { ...all, tokens: { from: 1, to: 2 } };
  for (const f of ranged.trace.activity.frames) {
    if (f.role === 'ffn-gate') { f.tokenStart = 1; f.tokenIds = [20]; f.fullShape = [1, 3]; f.shape = [3]; f.values = f.values.slice(3); }
  }
  assert.deepEqual(missingReason(ranged, 'ffn-gate', 0), { kind: 'not-selected', nearest: 1, text: 'This recording kept tokens 1–1 for this view.' });

  const layered = structuredClone(recording);
  layered.trace.activity.selection = { ...all, layers: [0] };
  assert.deepEqual(missingReason(layered, 'ffn-gate', 0, 'overview', 1), { kind: 'not-selected', nearest: 0, text: 'This recording kept layer 1 only.' });
  assert.equal(missingReason(layered, 'ffn-gate', 0, 'overview', 0).kind, 'none', 'a recorded layer is complete');
  assert.equal(missingReason(layered, 'ffn-gate', 0).kind, 'not-selected', 'with every layer shown, the ones left out are explained');
  assert.equal(missingReason(layered, 'difference', 0).kind, 'none', 'layer outputs are always recorded, so the selection does not apply to them');

  const ownLimit = structuredClone(recording);
  for (const f of ownLimit.trace.activity.frames) {
    if (f.role === 'kv-key') { f.status = 'dropped'; f.values = null; f.metadata.limit = 'activityRoleMaxBytes'; }
  }
  ownLimit.trace.activity.coverage['kv-key'].maxBytes = 2 ** 21;
  assert.match(missingReason(ownLimit, 'kv-new', 1).text, /This view reached its own size limit \(2 MB\) at token 0/);

  const malformed = structuredClone(recording);
  malformed.trace.activity.selection = { ...all, layers: [99] };
  assert.throws(() => validateRecording(malformed), /Invalid or oversized model activity recording/);
  console.log('Selective capture: kinds, layers, token ranges, per-kind budgets, one-time fixed values, coverage counts and viewer reasons passed.');
}

// Display theme changes colors only, with no alpha data or numerical rescaling.
const light = colorGrid([-1, 0, 1, undefined, NaN], 1, 'signed');
const dark = colorGrid([-1, 0, 1, undefined, NaN], 1, 'signed', true);
assert.deepEqual(heatmapColor(0, 1, 'signed', true), heatmapColors('signed', true)[1]);
assert.notDeepEqual(dark.rgba.slice(4, 7), light.rgba.slice(4, 7));
assert.notDeepEqual(dark.rgba.slice(4, 7), dark.rgba.slice(12, 15), 'Zero is distinct from a missing capture');
assert.deepEqual(dark.rgba.slice(12), light.rgba.slice(12), 'Missing, invalid and padding semantics survive theme changes');
assert([...dark.rgba].filter((_, i) => i % 4 === 3).every(a => a === 255));
