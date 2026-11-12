// Contract: doppler.routing-aware-acceptance/v1 accepts an unforced routed-MoE run
// only when its routing divergences reduce to one near-tie root plus that root's causal
// cascade, the forced-indices control and same-input checks pass the unchanged
// quantized-v1 tolerance, pre-registered task-quality limits pass, and every artifact
// names the qualified tuple. Synthetic fixtures; no GPU.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import {
  attributeDivergences,
  evaluateRoutingAwareAcceptance,
  extractAnswer,
  hashPolicy,
  samplePositions,
} from '../../tools/lib/routing-aware-acceptance.js';
import { runEvaluation } from '../../tools/evaluate-routing-aware-acceptance.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const readRepoJson = async relative => JSON.parse(await fs.readFile(path.join(ROOT, relative), 'utf8'));
const checkedInPolicy = await readRepoJson('tools/policies/routing-aware-acceptance.json');
const tolerancePolicies = (await readRepoJson('src/config/evidence/boundary-evidence-policy.json')).tolerancePolicies;

// The checked-in policy binds the checked-in prompt set and is not yet approved.
const pinnedPrompts = await readRepoJson(checkedInPolicy.taskQuality.promptSet.path);
assert.equal(computeCanonicalJsonSha256(pinnedPrompts), checkedInPolicy.taskQuality.promptSet.sha256);
assert.equal(checkedInPolicy.approval.status, 'proposed');

const GEOMETRY = { numLayers: 3, numTokens: 4, numExperts: 6, topK: 2 };
// CPU row: experts 0 and 1 selected; expert 2 trails expert 1 by about 1e-5 (near-tie),
// expert 5 trails by 7 (decisive).
const CPU_ROW = [5, 4.00001, 4, 1, 0, -3];
const MANIFEST = 'a'.repeat(64);
const PRODUCER = checkedInPolicy.cpuReference.producerSha256;
const TUPLE = {
  modelId: 'synthetic-routed-moe', artifactManifestSha256: MANIFEST,
  precision: { runtimeProfile: 'profiles/synthetic', weights: 'q4k', activations: 'f32', kvCache: 'f32', routerLogits: 'f32' },
  backend: { surface: 'node', api: 'webgpu', binding: 'webgpu@0.4.0' },
  device: { vendor: 'amd', architecture: 'rdna3', adapter: 'synthetic adapter' },
};
const PROMPTS = {
  schema: 'doppler.routing-aware-prompt-set/v1', id: 'synthetic', cases: [
    { id: 'c0', kind: 'short-arithmetic', text: 'two plus two is', expected: '4', minimumTokens: 1, maximumTokens: 64 },
  ],
  generation: { temperature: 0, topK: 1, topP: 1, maxNewTokens: 4, stopAtEos: true },
  answerExtraction: 'first-whitespace-token-strip-trailing-punctuation',
  samplePositions: { stride: 2, offset: 1, includeLast: true },
};
const APPROVED_AT = '2026-11-13T00:00:00Z';
const RUN_AT = '2026-11-14T00:00:00Z';

function approvedPolicy(promptSet = PROMPTS) {
  const policy = structuredClone(checkedInPolicy);
  policy.approval = { ...policy.approval, status: 'approved', approvedBy: 'owner', approvedAtUtc: APPROVED_AT };
  policy.nearTie.status = 'approved';
  policy.taskQuality.status = 'approved';
  policy.taskQuality.promptSet.sha256 = computeCanonicalJsonSha256(promptSet);
  const { adapter, ...device } = TUPLE.device;
  policy.qualificationTuples = [{ id: 'synthetic-tuple', ...TUPLE, device }];
  return policy;
}

// flips: [{ layer, token, experts }] executed by the GPU instead of the CPU set {0, 1}.
function routingArrays(flips) {
  const { numLayers, numTokens, numExperts, topK } = GEOMETRY;
  const cpuLogits = [], cpuExperts = [], gpuExperts = [];
  for (let layer = 0; layer < numLayers; layer++) {
    cpuLogits.push(Float32Array.from({ length: numTokens * numExperts }, (_, i) => CPU_ROW[i % numExperts]));
    cpuExperts.push(Uint32Array.from({ length: numTokens * topK }, (_, i) => i % topK));
    // Same set in a different slot order is not a divergence.
    gpuExperts.push(Uint32Array.from({ length: numTokens * topK }, (_, i) => 1 - (i % topK)));
  }
  for (const { layer, token, experts } of flips) gpuExperts[layer].set(experts, token * topK);
  return { cpuLogits, cpuExperts, gpuExperts };
}

function sameInputChecks() {
  const all = [0, 1, 2];
  const check = (family, layers) => ({ family, caseId: 'c0', layers, count: 128, outsidePolicy: 0 });
  return { tuple: TUPLE, policyId: 'doppler.boundary-tolerance/quantized-v1', checks: [
    check('token-embedding', []), check('hyper-connection', all), check('gated-delta', [0, 2]),
    check('full-attention', [1]), check('routed-moe', all), check('shared-expert', all),
    check('final-norm', []), check('lm-head', []),
  ] };
}

function taskQuality(policy) {
  return {
    schema: 'doppler.routing-aware-task-quality/v1', tuple: TUPLE, policySha256: hashPolicy(policy),
    promptSetSha256: policy.taskQuality.promptSet.sha256, runStartedAtUtc: RUN_AT,
    caseTokenCounts: { c0: GEOMETRY.numTokens },
    distributionRows: [1, 3].map(position => ({ caseId: 'c0', position, cpuTop1: 19, gpuTop1: 19,
      cpuTop1MarginLogits: 2.5, klCpuGpuNats: 1e-5 })),
    answers: [{ caseId: 'c0', cpuTokens: [220, 19], gpuTokens: [220, 19], cpuText: ' 4.', gpuText: ' 4.' }],
  };
}

function bundle({ flips = [], policy = approvedPolicy(), ownDifferences = [], forcedViolations = 0, tuple = TUPLE } = {}) {
  const { cpuLogits, cpuExperts, gpuExperts } = routingArrays(flips);
  const forcedLayers = Array.from({ length: GEOMETRY.numLayers }, (_, layer) => ({
    layer, routerLogitsMaxAbsDiffVsReference: 1e-4,
    ownSelectionDifferences: ownDifferences.filter(entry => entry.layer === layer)
      .map(({ token, experts }) => ({ token, experts })),
  }));
  return {
    schema: 'doppler.routing-aware-acceptance-evidence/v1', policySha256: hashPolicy(policy),
    promptSetSha256: policy.taskQuality.promptSet.sha256, runStartedAtUtc: RUN_AT, tuple,
    cases: [{
      caseId: 'c0', ...GEOMETRY,
      cpu: { producerSha256: PRODUCER, artifactManifestSha256: MANIFEST, logits: cpuLogits, experts: cpuExperts },
      unforced: { mode: 'observe', selectionSource: 'gpu-executed-readback', tuple, experts: gpuExperts,
        prefillLogitsSha256: 'b'.repeat(64), uninstrumentedPrefillLogitsSha256: 'b'.repeat(64) },
      forcedIndices: { mode: 'indices', completed: true, tuple, executedSelectionDifferingTokenLayers: 0,
        logits: { policyId: 'doppler.boundary-tolerance/quantized-v1', count: 1000, outsidePolicy: forcedViolations },
        layers: forcedLayers },
    }],
    sameInputOperators: { ...sameInputChecks(), tuple },
    taskQuality: { ...taskQuality(policy), tuple },
  };
}

function evaluate(evidence, { policy = approvedPolicy(), promptSet = PROMPTS } = {}) {
  return evaluateRoutingAwareAcceptance({ policy, promptSet, tolerancePolicies, evidence });
}
const codes = verdict => new Set(verdict.reasons.map(entry => entry.code));

// Helpers.
assert.equal(extractAnswer(' 353 units.'), '353');
assert.equal(extractAnswer('\nlantern.'), 'lantern');
assert.deepEqual(samplePositions({ stride: 64, offset: 63, includeLast: true }, 130), [63, 127, 129]);
assert.deepEqual(samplePositions({ stride: 2, offset: 1, includeLast: true }, 4), [1, 3]);
assert.deepEqual(attributeDivergences([{ layer: 1, token: 1 }, { layer: 2, token: 1 }, { layer: 2, token: 3 }]),
  { roots: [{ layer: 1, token: 1 }], cascades: 2 });
// Same layer and earlier tokens are outside the cone.
assert.deepEqual(attributeDivergences([{ layer: 1, token: 2 }, { layer: 1, token: 3 }, { layer: 2, token: 1 }]).roots,
  [{ layer: 1, token: 2 }, { layer: 1, token: 3 }, { layer: 2, token: 1 }]);

// 1. Clean pass: identical routing (slot order ignored), controls and task quality pass.
{
  const verdict = evaluate(bundle());
  assert.deepEqual(verdict.reasons, []);
  assert.equal(verdict.verdict, 'pass');
  assert.equal(verdict.qualifiedTuple.qualificationTupleId, 'synthetic-tuple');
  assert.equal(verdict.qualifiedTuple.device.adapter, 'synthetic adapter');
  assert.equal(verdict.cases[0].routing.divergentTokenLayers, 0);
}

// 2. One near-tie flip at L1/t1 and its downstream cascade (decisive later flips) pass.
{
  const verdict = evaluate(bundle({ flips: [
    { layer: 1, token: 1, experts: [0, 2] },
    { layer: 2, token: 1, experts: [0, 5] },
    { layer: 2, token: 3, experts: [5, 1] },
  ] }));
  assert.deepEqual(verdict.reasons, []);
  assert.equal(verdict.verdict, 'pass');
  const routing = verdict.cases[0].routing;
  assert.equal(routing.divergentTokenLayers, 3);
  assert.equal(routing.cascadeTokenLayers, 2);
  assert.equal(routing.roots.length, 1);
  assert.ok(routing.roots[0].nearTie);
  assert.ok(Math.abs(routing.roots[0].flipDisplacement - 1e-5) < 2e-6);
  assert.ok(Math.abs(routing.roots[0].bound - 2e-4) < 1e-12);
}

// 3. First divergence at a large-margin choice fails.
{
  const verdict = evaluate(bundle({ flips: [{ layer: 1, token: 1, experts: [0, 5] }] }));
  assert.equal(verdict.verdict, 'fail');
  assert.ok(codes(verdict).has('ROOT_NOT_NEAR_TIE'));
}

// 4. Two independent first divergences fail even when both are near-ties.
{
  const verdict = evaluate(bundle({ flips: [
    { layer: 1, token: 1, experts: [0, 2] },
    { layer: 2, token: 0, experts: [0, 2] },
  ] }));
  assert.equal(verdict.verdict, 'fail');
  assert.deepEqual([...codes(verdict)], ['INDEPENDENT_ROOTS_EXCEEDED']);
}

// 5. Forced-control violations fail: quantized-v1 logit violations, or a GPU own choice
// under forcing that is not a near-tie.
{
  const violating = evaluate(bundle({ forcedViolations: 3 }));
  assert.equal(violating.verdict, 'fail');
  assert.ok(codes(violating).has('FORCED_CONTROL_VIOLATIONS'));
  const nearTieOwn = evaluate(bundle({ ownDifferences: [{ layer: 0, token: 2, experts: [0, 2] }] }));
  assert.equal(nearTieOwn.verdict, 'pass');
  const decisiveOwn = evaluate(bundle({ ownDifferences: [{ layer: 0, token: 2, experts: [0, 4] }] }));
  assert.ok(codes(decisiveOwn).has('FORCED_CONTROL_OWN_FLIP_NOT_NEAR_TIE'));
}

// 6. Missing task-quality results (or a missing control) fail closed.
{
  const evidence = bundle();
  evidence.taskQuality = null;
  const verdict = evaluate(evidence);
  assert.equal(verdict.verdict, 'fail');
  assert.ok(codes(verdict).has('TASK_QUALITY_MISSING'));
  const noControl = bundle();
  noControl.cases[0].forcedIndices = null;
  assert.ok(codes(evaluate(noControl)).has('INPUT_MISSING'));
  const quality = bundle();
  quality.taskQuality.distributionRows.pop();
  assert.ok(codes(evaluate(quality)).has('TASK_QUALITY_ROWS_MISSING'));
}

// 7. A different model, precision or backend tuple fails, whole-bundle or per artifact.
{
  for (const change of [
    tuple => { tuple.modelId = 'other-model'; },
    tuple => { tuple.precision.kvCache = 'f16'; },
    tuple => { tuple.backend.surface = 'browser'; },
  ]) {
    const tuple = structuredClone(TUPLE);
    change(tuple);
    const verdict = evaluate(bundle({ tuple }));
    assert.equal(verdict.verdict, 'fail');
    assert.ok(codes(verdict).has('TUPLE_NOT_QUALIFIABLE'));
  }
  const mixed = bundle();
  mixed.taskQuality.tuple = { ...TUPLE, device: { ...TUPLE.device, adapter: 'another adapter' } };
  assert.ok(codes(evaluate(mixed)).has('TUPLE_MISMATCH'));
}

// Policy approval and freshness: the checked-in proposed policy never passes, and a
// run that started before approval is not fresh.
{
  const proposed = structuredClone(approvedPolicy());
  proposed.approval.status = 'proposed';
  const verdict = evaluate(bundle({ policy: proposed }), { policy: proposed });
  assert.ok(codes(verdict).has('POLICY_NOT_APPROVED'));
  const stale = bundle();
  stale.runStartedAtUtc = '2026-11-12T00:00:00Z';
  assert.ok(codes(evaluate(stale)).has('RUN_NOT_FRESH'));
  const loosened = { ...tolerancePolicies, 'doppler.boundary-tolerance/quantized-v1': { absolute: 0.1, relative: 0.05, requireDtypeMatch: false } };
  const policy = approvedPolicy();
  assert.ok(codes(evaluateRoutingAwareAcceptance({ policy, promptSet: PROMPTS, tolerancePolicies: loosened,
    evidence: bundle({ policy }) })).has('TOLERANCE_CHANGED'));
}

// File loader: hashes are verified and a missing file fails closed.
{
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'routing-aware-acceptance-'));
  try {
    const policy = approvedPolicy();
    const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
    const write = async (name, bytes) => {
      await fs.mkdir(path.dirname(path.join(dir, name)), { recursive: true });
      await fs.writeFile(path.join(dir, name), bytes);
      return { file: path.basename(name), sha256: sha(bytes) };
    };
    const writeJson = (name, value) => write(name, Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
    await writeJson(policy.taskQuality.promptSet.path, PROMPTS);
    await writeJson(policy.numericalTolerance.policyPath, { tolerancePolicies });
    const policyPath = path.join(dir, 'policy.json');
    await fs.writeFile(policyPath, JSON.stringify(policy));
    const memory = bundle({ policy, flips: [{ layer: 1, token: 1, experts: [0, 2] }] });
    const item = memory.cases[0];
    const bytesOf = array => Buffer.from(array.buffer);
    const files = [];
    for (let layer = 0; layer < GEOMETRY.numLayers; layer++) {
      files.push({ layer,
        experts: await write(`cpu/experts-${layer}.u32`, bytesOf(item.cpu.experts[layer])),
        logits: await write(`cpu/logits-${layer}.f32`, bytesOf(item.cpu.logits[layer])) });
    }
    await writeJson('cpu/fixture.json', { schema: 'doppler.flash-next-routing-selection-fixture/v1', sequenceId: 'c0',
      layers: GEOMETRY.numLayers, numTokens: GEOMETRY.numTokens, numExperts: GEOMETRY.numExperts, topK: GEOMETRY.topK,
      manifestSha256: MANIFEST, producerSha256: PRODUCER, files });
    const gpuLayers = [];
    for (let layer = 0; layer < GEOMETRY.numLayers; layer++) {
      gpuLayers.push({ layer, experts: await write(`gpu-experts-${layer}.u32`, bytesOf(item.unforced.experts[layer])) });
    }
    const { experts: _gpu, ...unforced } = item.unforced;
    const quality = await writeJson('task-quality.json', memory.taskQuality);
    const onDisk = { ...memory, taskQuality: quality, cases: [{ ...item, cpu: { fixture: 'cpu/fixture.json' },
      unforced: { ...unforced, layers: gpuLayers } }] };
    const evidencePath = path.join(dir, 'evidence.json');
    await fs.writeFile(evidencePath, JSON.stringify(onDisk));
    const passed = await runEvaluation({ evidencePath, policyPath, repoRoot: dir });
    assert.deepEqual(passed.reasons, []);
    assert.equal(passed.verdict, 'pass');

    await fs.writeFile(path.join(dir, 'gpu-experts-2.u32'), Buffer.alloc(GEOMETRY.numTokens * GEOMETRY.topK * 4));
    const tampered = await runEvaluation({ evidencePath, policyPath, repoRoot: dir });
    assert.equal(tampered.verdict, 'fail');
    assert.ok(codes(tampered).has('INPUT_UNREADABLE'));

    await fs.rm(path.join(dir, 'task-quality.json'));
    const missing = await runEvaluation({ evidencePath, policyPath, repoRoot: dir });
    assert.ok(codes(missing).has('TASK_QUALITY_MISSING'));
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}

console.log('routing-aware-acceptance.test: ok');
