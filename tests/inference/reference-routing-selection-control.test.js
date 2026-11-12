import assert from 'node:assert/strict';
import {
  MASKED_ROUTER_LOGIT,
  compareRouteSelections,
  countOwnSelectionDifferences,
  installReferenceRoutingSelectionControl,
  maskRouterLogits,
  referenceWeightsInObservedOrder,
} from '../helpers/reference-routing-selection-control.js';

const geometry = { numTokens: 2, numExperts: 4, topK: 2 };
const logits = Float32Array.from([0.9, 0.5, 0.49, -1, 0.1, 0.2, 0.3, 0.4]);
// Token 0: the reference picks expert 2 where the GPU's own logits pick expert 1.
const experts = Uint32Array.from([0, 2, 3, 2]);
const scores = Float32Array.from([0.6, 0.4, 0.55, 0.45]);

const masked = maskRouterLogits(logits, experts, geometry);
assert.deepEqual(Array.from(masked), [
  logits[0], MASKED_ROUTER_LOGIT, logits[2], MASKED_ROUTER_LOGIT,
  MASKED_ROUTER_LOGIT, MASKED_ROUTER_LOGIT, logits[6], logits[7],
]);
assert.throws(() => maskRouterLogits(logits, Uint32Array.from([0, 0, 3, 2]), geometry), /duplicate reference expert/);
assert.throws(() => maskRouterLogits(logits, Uint32Array.from([0, 4, 3, 2]), geometry), /out of range/);

assert.deepEqual(countOwnSelectionDifferences(logits, experts, geometry), { differingTokens: 1, firstToken: 0 });
assert.deepEqual(countOwnSelectionDifferences(masked, experts, geometry), { differingTokens: 0, firstToken: null });

// Same sets in a different slot order compare by expert ID.
const observed = Uint32Array.from([2, 0, 3, 2]);
const observedWeights = Float32Array.from([0.4, 0.6, 0.5, 0.5]);
const compared = compareRouteSelections(observed, observedWeights, experts, scores, geometry);
assert.equal(compared.differingTokens, 0);
assert.ok(Math.abs(compared.maxWeightAbsDiff - 0.05) < 1e-6);
assert.deepEqual(Array.from(referenceWeightsInObservedOrder(observed, experts, scores, geometry)),
  [scores[1], scores[0], scores[2], scores[3]]);
const wrong = Uint32Array.from([1, 0, 3, 2]);
assert.equal(compareRouteSelections(wrong, observedWeights, experts, scores, geometry).firstToken, 0);
assert.throws(() => referenceWeightsInObservedOrder(wrong, experts, scores, geometry), /not in the reference set/);

// Installation patches only the router method and restores it; unknown modes fail.
class FakeRouter { async computeRouterLogitsGPU() { return 'original'; } }
const original = FakeRouter.prototype.computeRouterLogitsGPU;
const fixture = { ...geometry, receiptSha256: 'f', perLayer: [{ layer: 0, experts, scores, logits }] };
assert.throws(() => installReferenceRoutingSelectionControl(fixture, 'logits', { MoERouter: FakeRouter }), /mode must be one of/);
const installed = installReferenceRoutingSelectionControl(fixture, 'observe', { MoERouter: FakeRouter });
assert.notEqual(FakeRouter.prototype.computeRouterLogitsGPU, original);
assert.equal(installed.control.normalModelQualified, false);
// Calls with another token count (a different prompt) pass through untouched.
assert.equal(await new FakeRouter().computeRouterLogitsGPU(null, 3, null, {}), 'original');
installed.restore();
assert.equal(FakeRouter.prototype.computeRouterLogitsGPU, original);

console.log('reference-routing-selection-control.test: ok');
