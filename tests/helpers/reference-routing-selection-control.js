import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

// Counterfactual TEST PORT for Flash-Next qualification diagnostics only. It is
// never imported by src/, never part of production execution, and a run that
// installs it is never normal model qualification. It forces the MoE top-k
// expert selection to an independently produced CPU source fixture:
// - 'observe': no replacement; reads the GPU's own top-k back and compares.
// - 'indices': GPU router logits outside the reference expert set become
//   -FLT_MAX, so the unchanged top-k shader selects exactly the reference set
//   and computes its weights from the GPU's own logits.
// - 'indices-weights': as 'indices', then the selected weights are replaced by
//   the reference weights, matched by expert ID.
export const ROUTING_SELECTION_MODES = Object.freeze(['observe', 'indices', 'indices-weights']);
export const MASKED_ROUTER_LOGIT = -3.4028234663852886e38;

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const view = (bytes, Type) => new Type(bytes.buffer, bytes.byteOffset, bytes.byteLength / Type.BYTES_PER_ELEMENT);

export function maskRouterLogits(logits, experts, { numTokens, numExperts, topK }) {
  assert.equal(logits.length, numTokens * numExperts);
  assert.equal(experts.length, numTokens * topK);
  const masked = new Float32Array(logits.length).fill(MASKED_ROUTER_LOGIT);
  for (let token = 0; token < numTokens; token++) {
    const seen = new Set();
    for (let slot = 0; slot < topK; slot++) {
      const expert = experts[token * topK + slot];
      assert.ok(expert < numExperts, `reference expert ${expert} out of range`);
      assert.ok(!seen.has(expert), `token ${token}: duplicate reference expert ${expert}`);
      seen.add(expert);
      const index = token * numExperts + expert;
      assert.ok(logits[index] > MASKED_ROUTER_LOGIT, 'reference expert logit must exceed the mask');
      masked[index] = logits[index];
    }
  }
  return masked;
}

function sortedSet(values, offset, topK) {
  return Array.from(values.subarray(offset, offset + topK)).sort((a, b) => a - b);
}

// Ordered top-k ranking of one logits row (descending, lower index wins ties,
// matching softmax_topk_strided.wgsl). Used only to report where the GPU's own
// selection would have differed from the reference under forcing.
function rankTopK(logits, offset, numExperts, topK) {
  const order = Array.from({ length: numExperts }, (_, expert) => expert);
  order.sort((a, b) => (logits[offset + b] - logits[offset + a]) || (a - b));
  return order.slice(0, topK);
}

export function countOwnSelectionDifferences(logits, experts, { numTokens, numExperts, topK }) {
  let differingTokens = 0, firstToken = null;
  for (let token = 0; token < numTokens; token++) {
    const own = rankTopK(logits, token * numExperts, numExperts, topK).sort((a, b) => a - b);
    const reference = sortedSet(experts, token * topK, topK);
    if (own.some((expert, i) => expert !== reference[i])) {
      differingTokens++;
      firstToken ??= token;
    }
  }
  return { differingTokens, firstToken };
}

export function compareRouteSelections(indices, weights, experts, scores, { numTokens, topK }) {
  let differingTokens = 0, firstToken = null, maxWeightAbsDiff = 0;
  for (let token = 0; token < numTokens; token++) {
    const base = token * topK;
    const observed = sortedSet(indices, base, topK);
    const reference = sortedSet(experts, base, topK);
    if (observed.some((expert, i) => expert !== reference[i])) {
      differingTokens++;
      firstToken ??= token;
      continue;
    }
    for (let slot = 0; slot < topK; slot++) {
      const referenceSlot = Array.prototype.indexOf.call(experts.subarray(base, base + topK), indices[base + slot]);
      maxWeightAbsDiff = Math.max(maxWeightAbsDiff, Math.abs(weights[base + slot] - scores[base + referenceSlot]));
    }
  }
  return { differingTokens, firstToken, maxWeightAbsDiff };
}

export function referenceWeightsInObservedOrder(indices, experts, scores, { numTokens, topK }) {
  const output = new Float32Array(numTokens * topK);
  for (let token = 0; token < numTokens; token++) {
    const base = token * topK;
    const reference = experts.subarray(base, base + topK);
    for (let slot = 0; slot < topK; slot++) {
      const referenceSlot = Array.prototype.indexOf.call(reference, indices[base + slot]);
      assert.ok(referenceSlot >= 0, `token ${token}: selected expert ${indices[base + slot]} is not in the reference set`);
      output[base + slot] = scores[base + referenceSlot];
    }
  }
  return output;
}

export async function loadRoutingSelectionFixture(directory, { manifestSha256, inputsSha256, referenceSha256 }) {
  const receiptBytes = await fs.readFile(path.join(directory, 'fixture.json'));
  const receipt = JSON.parse(receiptBytes);
  assert.equal(receipt.schema, 'doppler.flash-next-routing-selection-fixture/v1');
  assert.equal(receipt.manifestSha256, manifestSha256, 'fixture artifact identity');
  assert.equal(receipt.inputsSha256, inputsSha256, 'fixture inputs identity');
  assert.equal(receipt.referenceSha256, referenceSha256, 'fixture reference identity');
  const { numTokens, numExperts, topK, layers } = receipt;
  assert.deepEqual(receipt.files.map(item => item.layer), Array.from({ length: layers }, (_, i) => i));
  const read = async (item, Type, width) => {
    const bytes = await fs.readFile(path.join(directory, item.file));
    assert.equal(sha256(bytes), item.sha256, `${item.file} hash`);
    assert.equal(bytes.length, numTokens * width * 4, `${item.file} size`);
    return view(bytes, Type);
  };
  const perLayer = [];
  for (const item of receipt.files) {
    const experts = await read(item.experts, Uint32Array, topK);
    const scores = await read(item.scores, Float32Array, topK);
    const logits = await read(item.logits, Float32Array, numExperts);
    assert.ok(scores.every(Number.isFinite) && logits.every(Number.isFinite));
    perLayer.push({ layer: item.layer, experts, scores, logits });
  }
  return { receipt, receiptSha256: sha256(receiptBytes), numTokens, numExperts, topK, perLayer };
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

// `memory` is the snapshot's buffer-pool module and `device` its GPU device, so
// the control uses the same module instances as the pipeline under test.
export function installReferenceRoutingSelectionControl(fixture, mode, { MoERouter, memory, getDevice }) {
  assert.ok(ROUTING_SELECTION_MODES.includes(mode), `mode must be one of ${ROUTING_SELECTION_MODES.join(', ')}`);
  const { numTokens, numExperts, topK } = fixture;
  const control = { kind: 'counterfactual-reference-routing-selection', mode,
    scope: mode === 'observe'
      ? 'GPU top-k read back and compared with the CPU source selection; no replacement'
      : 'Test-only forced CPU source expert selection; NOT normal learned-model qualification',
    normalModelQualified: false, fixtureSha256: fixture.receiptSha256, routerLayers: [], topKLayers: [] };
  const original = MoERouter.prototype.computeRouterLogitsGPU;
  const applies = numTokensSeen => numTokensSeen === numTokens;
  MoERouter.prototype.computeRouterLogitsGPU = async function(input, count, gpuContext, options) {
    const output = await original.call(this, input, count, gpuContext, options);
    if (!applies(count)) return output;
    assert.equal(options.recorder ?? null, null, 'routing control requires the declared immediate lane');
    assert.equal(options.outputDtype, 'f32');
    assert.equal(this.numExperts, numExperts);
    const reference = fixture.perLayer[options.layerIdx];
    assert.ok(reference, `no reference routing for layer ${options.layerIdx}`);
    const bytes = numTokens * numExperts * 4;
    const logits = new Float32Array(await memory.readBuffer(output, bytes));
    assert.ok(logits.every(Number.isFinite), 'control must not hide nonfinite GPU routing');
    const own = countOwnSelectionDifferences(logits, reference.experts, { numTokens, numExperts, topK });
    const row = { layer: options.layerIdx, gpuLogitsSha256: sha256(new Uint8Array(logits.buffer)),
      routerLogitsMaxAbsDiffVsReference: maxAbsDiff(logits, reference.logits),
      ownSelectionDifferingTokens: own.differingTokens, ownSelectionFirstDifferingToken: own.firstToken,
      replaced: false };
    if (mode !== 'observe') {
      const masked = maskRouterLogits(logits, reference.experts, { numTokens, numExperts, topK });
      getDevice().queue.writeBuffer(output, 0, masked);
      const written = new Uint8Array(await memory.readBuffer(output, bytes));
      assert.equal(sha256(written), sha256(new Uint8Array(masked.buffer)), 'masked logits write');
      row.replaced = true;
    }
    control.routerLayers.push(row);
    return output;
  };
  async function observeTopK(record) {
    if (!applies(record.numTokens)) return;
    assert.equal(record.topK, topK);
    assert.equal(record.activationDtype, 'f32');
    const reference = fixture.perLayer[record.layer];
    const size = numTokens * topK * 4;
    const indices = new Uint32Array(await memory.readBuffer(record.indicesBuffer, size));
    let weights = new Float32Array(await memory.readBuffer(record.weightsBuffer, size));
    const before = compareRouteSelections(indices, weights, reference.experts, reference.scores, { numTokens, topK });
    const row = { layer: record.layer, selectedSetDifferingTokens: before.differingTokens,
      firstDifferingToken: before.firstToken, selectedWeightMaxAbsDiff: before.maxWeightAbsDiff, weightsReplaced: false };
    if (mode !== 'observe') assert.equal(before.differingTokens, 0, `layer ${record.layer}: forced selection not applied`);
    if (mode === 'indices-weights') {
      const replacement = referenceWeightsInObservedOrder(indices, reference.experts, reference.scores, { numTokens, topK });
      getDevice().queue.writeBuffer(record.weightsBuffer, 0, replacement);
      weights = new Float32Array(await memory.readBuffer(record.weightsBuffer, size));
      assert.equal(sha256(new Uint8Array(weights.buffer)), sha256(new Uint8Array(replacement.buffer)), 'weights write');
      row.weightsReplaced = true;
      row.selectedWeightMaxAbsDiffAfter = compareRouteSelections(indices, weights, reference.experts, reference.scores,
        { numTokens, topK }).maxWeightAbsDiff;
    }
    control.topKLayers.push(row);
  }
  return { control, observeTopK, restore() { MoERouter.prototype.computeRouterLogitsGPU = original; } };
}
