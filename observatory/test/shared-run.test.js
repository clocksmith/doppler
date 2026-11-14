import test from 'node:test';
import assert from 'node:assert/strict';
import { alternatives, compareRuns, createRun, describeCaptures, describeComparison, findParent, forest, forkRequest, layerVector, serializeRun, stepTimings, validateRun } from '../public/shared-run.js';

const LAYERS = 2;
const WIDTH = 3;
// A receipt shaped like Doppler's: 2 prompt tokens, then an answer.
function receipt({ answer = [7, 8, 9], bump = null, policy = {}, steps = null, inputTokenIds = [1, 2], withLayers = true, withProbabilities = true } = {}) {
  const value = (position, layer, i) => position + layer / 10 + i / 100 + (bump && bump.position === position && bump.layer <= layer ? bump.by : 0);
  const frame = (tokenStart, count, layer) => ({ role: 'layer-output', status: 'captured', layer, tokenStart, tokenIds: new Array(count).fill(0),
    values: Array.from({ length: count * WIDTH }, (_, k) => value(tokenStart + Math.floor(k / WIDTH), layer, k % WIDTH)) });
  const frames = [];
  for (let layer = 0; layer < LAYERS; layer++) frames.push(frame(0, inputTokenIds?.length ?? 2, layer));
  answer.slice(0, -1).forEach((_, i) => { for (let layer = 0; layer < LAYERS; layer++) frames.push(frame(2 + i, 1, layer)); });
  return {
    policy: { id: 'demo/activation-grid', modifiesExecution: true, performanceRepresentative: false, gpuTimestampQueries: Boolean(steps), ...policy },
    fingerprint: { identity: { artifact: { modelId: 'm', manifestHash: 'sha256:aa' }, tokenizer: { digest: 'sha256:tt' } } },
    inputTokenIds, generatedTokenIds: answer, outputText: answer.join(' '), wallTimingMs: 120, performanceRepresentative: false,
    tokens: withProbabilities ? answer.map((tokenId, index) => ({ index, tokenId, text: ` t${tokenId}`, probability: 0.6,
      topCandidates: [{ tokenId, text: ` t${tokenId}`, probability: 0.6 }, { tokenId: tokenId + 100, text: ` u${tokenId}`, probability: 0.3 }] })) : [],
    generationEvidence: {
      generationConfig: { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, maxTokens: 3, useChatTemplate: true, seed: null },
      backendIdentity: { adapter: { vendor: 'v', architecture: 'a' }, activationDtype: 'f32', kernelPathId: 'k' },
      stats: { ttftMs: 20, decodeTimeMs: 100, decodeTokens: answer.length - 1, prefillTimeMs: 18, prefillTokens: 2, stopReason: 'max-tokens',
        gpuTimePrefillMs: null, gpuTimeDecodeMs: steps ? 9 : null, decodeProfileSteps: steps ?? [] },
    },
    activationTrace: withLayers ? { geometry: { layers: LAYERS, width: WIDTH }, frames } : undefined,
  };
}
const make = (options = {}, extra = {}) => createRun({ receipt: receipt(options), prompt: 'p', environment: { runtimeVersion: '1.0.0', userAgent: 'ua' }, createdAt: '2026-11-10T00:00:00.000Z', ...extra });

test('a run keeps its answer, settings, hardware, timing and layer outputs under one identity', async () => {
  const run = await make();
  assert.match(run.id, /^run-[0-9a-f]{16}$/);
  assert.equal(run.model.manifestHash, 'sha256:aa');
  assert.deepEqual(run.tokens.map((t) => [t.tokenId, t.text, t.probability]), [[7, ' t7', 0.6], [8, ' t8', 0.6], [9, ' t9', 0.6]]);
  assert.equal(run.timing.representative, false);
  assert.equal(run.timing.gpu, null);
  assert.deepEqual(run.layerOutputs.positions, [0, 1, 2, 3]);
  assert.deepEqual([...layerVector(run, 3, 1)].map((v) => +v.toFixed(2)), [3.1, 3.11, 3.12]);
  assert.equal(layerVector(run, 4, 0), null, 'a position that was not recorded has no vector');
  assert.notEqual((await make({}, { createdAt: '2026-11-10T00:00:01.000Z' })).id, run.id, 'a second recording is a different run');
});

test('a saved run opens again without anything else, and a damaged one says why', async () => {
  const run = await make();
  layerVector(run, 0, 0);
  const reopened = validateRun(JSON.parse(serializeRun(run)));
  assert.equal(reopened.layerOutputs.decoded, undefined, 'working state is not written to the file');
  assert.deepEqual([...layerVector(reopened, 3, 1)], [...layerVector(run, 3, 1)]);
  assert.throws(() => validateRun({ schema: 'other' }), /its format is "other"/);
  assert.throws(() => validateRun({ name: 'not a run' }), /does not say what it holds/);
  const cut = JSON.parse(serializeRun(run));
  cut.layerOutputs.data = cut.layerOutputs.data.slice(8);
  assert.throws(() => validateRun(cut), /do not match their declared size/);
  const bad = JSON.parse(serializeRun(run));
  bad.tokens[1].probability = 7;
  assert.throws(() => validateRun(bad), /answer token 1 is malformed/);
});

test('alternatives show what was taken, what was run, and the probability left unlisted', async () => {
  const trunk = await make();
  const request = forkRequest(trunk, 1, 108);
  const branch = await createRun({ receipt: receipt({ answer: [55], inputTokenIds: request.inputTokenIds }), prompt: 'p', parent: request.parent });
  const view = alternatives(trunk, 1, [trunk, branch]);
  assert.deepEqual(view.candidates.map((c) => [c.tokenId, c.state]), [[8, 'taken'], [108, 'run']]);
  assert.equal(view.candidates[1].branchRunId, branch.id);
  assert.ok(Math.abs(view.unlistedProbability - 0.1) < 1e-9, 'probability outside the list is reported, not dropped');
  assert.deepEqual(alternatives(trunk, 0, [trunk, branch]).candidates.map((c) => c.state), ['taken', 'not-run']);
  assert.equal(alternatives(await make({ withProbabilities: false }), 0), null);
});

test('a branch replays the exact tokens before the fork and keeps the same sampling and length', async () => {
  const trunk = await make();
  const request = forkRequest(trunk, 1, 108);
  assert.deepEqual(request.inputTokenIds, [1, 2, 7, 108]);
  assert.deepEqual(request.generation, { maxTokens: 1, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, useChatTemplate: true });
  assert.deepEqual(request.parent, { runId: trunk.id, index: 1, replacedTokenId: 8, tokenId: 108, tokenText: ' u8', probability: 0.3 });
  assert.throws(() => forkRequest(trunk, 1, 8), /already took/);
  assert.throws(() => forkRequest(trunk, 2, 109), /no room to continue/);
  assert.throws(() => forkRequest(await_free(trunk), 0, 107), /did not keep the exact tokens/);
  const child = await createRun({ receipt: receipt({ answer: [55], inputTokenIds: request.inputTokenIds }), prompt: 'p', parent: request.parent });
  const tree = forest(trunk, [child, trunk]);
  assert.deepEqual([tree.run.id, tree.branches.length, tree.branches[0].run.id, tree.branches[0].branches.length], [trunk.id, 1, child.id, 0]);
});
function await_free(run) { return { ...run, inputTokenIds: null }; }

test('two runs that agree show no difference', async () => {
  const result = compareRuns(await make(), await make({}, { environment: { runtimeVersion: '2.0.0', userAgent: 'ua' } }));
  assert.equal(result.comparable, true);
  assert.equal(result.tokens.firstDifference, null);
  assert.deepEqual([result.layers.firstNonzero, result.layers.firstAboveTolerance], [null, null]);
  assert.deepEqual(result.circumstances, [{ field: 'Runtime version', a: '1.0.0', b: '2.0.0' }]);
  const preview = await make({}, { environment: { runtimeVersion: '1.0.0', runtimeChannel: 'preview', userAgent: 'ua' } });
  assert.deepEqual(compareRuns(await make({}, { environment: { runtimeVersion: '1.0.0', runtimeChannel: 'hosted', userAgent: 'ua' } }), preview).circumstances, [{ field: 'Runtime channel', a: 'hosted', b: 'preview' }]);
});

test('the first observed difference is the earliest token position, then the earliest layer', async () => {
  const a = await make();
  const b = await make({ bump: { position: 2, layer: 1, by: 0.5 } });
  const { layers } = compareRuns(a, b);
  assert.deepEqual([layers.firstAboveTolerance.position, layers.firstAboveTolerance.layer], [2, 1]);
  assert.equal(layers.relative[layers.positions.indexOf(2)][0], 0, 'the layer before it agrees');
  assert.ok(layers.relative[layers.positions.indexOf(2)][1] > 0.1);
  const small = compareRuns(a, await make({ bump: { position: 1, layer: 0, by: 1e-5 } }));
  assert.equal(small.layers.firstAboveTolerance, null, 'a difference under the tolerance is not called a difference');
  assert.deepEqual([small.layers.firstNonzero.position, small.layers.firstNonzero.layer], [1, 0], 'but it is still reported');
});

test('layers are compared only while both runs were given the same tokens', async () => {
  const result = compareRuns(await make(), await make({ answer: [7, 88, 9] }));
  assert.equal(result.tokens.firstDifference, 1);
  assert.deepEqual([result.tokens.a.tokenId, result.tokens.b.tokenId], [8, 88]);
  assert.deepEqual(result.layers.positions, [0, 1, 2], 'position 3 holds different tokens in the two runs');
});

test('runs that are not the same question are refused with the reason', async () => {
  const a = await make();
  const other = await make({ inputTokenIds: [1, 3] });
  assert.match(compareRuns(a, other).blockers[0], /different input tokens \(they first differ at input token 2\)/);
  const sampled = await make();
  sampled.sampling.temperature = 0.7;
  assert.match(compareRuns(a, sampled).blockers[0], /different sampling settings \(temperature\)/);
  const files = await make();
  files.model.manifestHash = 'sha256:bb';
  assert.equal(compareRuns(a, files).comparable, false);
});

test('runs without layer outputs are still compared by their tokens', async () => {
  const result = compareRuns(await make(), await make({ withLayers: false }));
  assert.equal(result.comparable, true);
  assert.equal(result.layers, null);
  assert.match(result.layersUnavailable, /did not record layer outputs/);
});

test('step timing sums kernels across layers and keeps unmeasured time unmeasured', async () => {
  const steps = [{ step: 1, timings: { 'matmul:qkv_proj:L0': 1, 'matmul:qkv_proj:L1': 2, attention: 4 } }, { step: 2, timings: { 'matmul:qkv_proj:L0': 1.5, attention: 3 } }];
  const timing = stepTimings(await make({ steps }));
  assert.equal(timing.available, true);
  assert.deepEqual(timing.steps.map((s) => [s.step, s.totalMs, s.tokenText]), [[1, 7, ' t8'], [2, 4.5, ' t9']]);
  assert.deepEqual(timing.kernels, [{ name: 'attention', ms: 7 }, { name: 'matmul:qkv_proj', ms: 4.5 }]);
  assert.equal(timing.prefillGpuMs, null, 'reading the prompt was not timed, which is not the same as zero');
  assert.match(stepTimings(await make()).reason, /without GPU timing/);
});

test('a branch finds its parent by exact tokens, and nothing else passes for one', async () => {
  const trunk = await make();
  const request = forkRequest(trunk, 1, 108);
  // Built the way the merged page meets it: a receipt with no parent stated.
  const orphan = await createRun({ receipt: receipt({ answer: [55], inputTokenIds: request.inputTokenIds }), prompt: 'p' });
  assert.equal(orphan.kind, 'answer');
  assert.deepEqual(findParent(orphan, [trunk, orphan]), { runId: trunk.id, index: 1, replacedTokenId: 8, tokenId: 108, tokenText: ' u8', probability: 0.3 });
  const adopted = await createRun({ receipt: receipt({ answer: [55], inputTokenIds: request.inputTokenIds }), prompt: 'p', parent: findParent(orphan, [trunk]) });
  assert.equal(adopted.kind, 'branch');
  // The same question asked again is not a branch, and neither is a longer prompt that merely starts the same way.
  assert.equal(findParent(await make({}, { createdAt: '2026-11-10T00:00:05.000Z' }), [trunk]), null);
  assert.equal(findParent(await make({ inputTokenIds: [1, 2, 7, 8] }), [trunk]), null, 'continuing with the token the trunk took is not a fork');
  assert.equal(findParent(await make({ inputTokenIds: [1, 2, 9, 108] }), [trunk]), null, 'the tokens before the fork must be the trunk\'s own');
  const otherFiles = await make({ inputTokenIds: request.inputTokenIds });
  otherFiles.model.manifestHash = 'sha256:bb';
  assert.equal(findParent(otherFiles, [trunk]), null, 'a run of other model files is not a branch of this one');
});

test('a run says what it recorded and what it left out', async () => {
  const bare = await make({ withLayers: false });
  assert.deepEqual(describeCaptures(bare), ['No internals were recorded: this run has its answer, settings and timing only.']);
  const layered = await make();
  assert.deepEqual(describeCaptures(layered), ['Layer outputs for 4 token positions × 2 layers.']);
  const base = receipt();
  base.activationTrace.activity = {
    recordedBytes: 2 ** 20, limits: { activityMaxBytes: 2 ** 21 }, unavailable: {},
    frames: [{ status: 'captured' }, { status: 'dropped' }, { status: 'dropped' }],
    selection: { roles: ['logits', 'kv-key'], layers: [0], tokens: { from: 2, to: 6 } },
    coverage: { logits: { dropped: 1, firstDroppedToken: 5 }, 'kv-key': { dropped: 1, firstDroppedToken: 7 }, 'ffn-gate': { dropped: 0, firstDroppedToken: null } },
  };
  const active = await createRun({ receipt: base, prompt: 'p' });
  assert.deepEqual(describeCaptures(active).slice(1), [
    'Activity: 1 captures, 1.0 MiB of a 2.0 MiB budget.',
    'Recorded by choice: only logits, kv-key.',
    'Recorded by choice: only layer 1.',
    'Recorded by choice: only tokens 2–5.',
    'Ran out of budget starting between tokens 5 and 7: logits, kv-key.',
  ]);
  assert.equal(JSON.parse(serializeRun(active)).captures.activity.dropped, 2, 'the account travels with the saved run');
});

test('a comparison of randomly sampled runs says that differences are expected', async () => {
  const a = await make();
  const b = await make({ answer: [7, 88, 9] });
  assert.equal(describeComparison(a, b).sampling, '');
  for (const run of [a, b]) run.sampling.temperature = 1;
  const view = describeComparison(a, b);
  assert.match(view.sampling, /sampled at random \(temperature 1, no fixed seed\)/);
  assert.equal(view.verdict, 'The runs differ.');
  assert.match(view.tokens, /first differ at token 2: " t8" \(60%\) in the first run, " t88" \(60%\) in the second/);
  assert.deepEqual([view.grid.columns.map((c) => c.label), view.grid.rows.length], [['i1', 'i2', 'a1'], 2]);
  const blocked = describeComparison(a, await make({ inputTokenIds: [1, 3] }));
  assert.deepEqual([blocked.comparable, blocked.grid, blocked.verdict], [false, null, 'These two runs are not the same question, so they cannot be compared.']);
});


test('detailed activity survives save and reopen without affecting layer-only recordings', async () => {
  const source = receipt();
  source.activationTrace.activity = { frames: [], weights: [], selection: { roles: ['kv-key'] } };
  source.activationTrace.frames[0].values = Float32Array.from(source.activationTrace.frames[0].values);
  const run = await createRun({ receipt: source, prompt: 'fixture' });
  assert.deepEqual(validateRun(JSON.parse(serializeRun(run))).activityTrace.frames[0].values, Array.from(source.activationTrace.frames[0].values));
  source.activationTrace.activity.frames.push({ role: 'changed-after-recording' });
  assert.equal(run.activityTrace.activity.frames.length, 0, 'receipt ownership is not shared');
  assert.equal((await make()).activityTrace, null, 'basic runs do not duplicate their layer trace');
  assert.throws(() => validateRun({ ...run, activityTrace: { geometry: {}, frames: 'bad' } }), /activity trace/);
});

test('imported candidates reject invalid probability and duplicate token identity', async () => {
  const run = await make();
  for (const probability of [-0.1, 1.1, Infinity]) {
    const damaged = JSON.parse(serializeRun(run));
    damaged.tokens[0].candidates[0].probability = probability;
    assert.throws(() => validateRun(damaged), /malformed/);
  }
  const duplicate = JSON.parse(serializeRun(run));
  duplicate.tokens[0].candidates.push(duplicate.tokens[0].candidates[0]);
  assert.throws(() => validateRun(duplicate), /malformed/);
});
