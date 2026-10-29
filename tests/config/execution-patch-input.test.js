import test from 'node:test';
import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { mergeExecutionPatchLists } from '../../src/config/merge/execution-patches.js';
import { applyExecutionPatch } from '../../src/inference/pipelines/text/execution-v1/validation.js';

test('runtime creation rejects mistyped patch actions before silently dropping requested kernel selections', () => {
  assert.throws(() => createDopplerConfig({ runtime: { inference: {
    executionPatch: { setKernel: [{ section: 'decode', op: 'q_proj', kernelKey: 'candidate' }] },
  } } }), /runtime\.inference\.executionPatch\.setKernel is unknown/);
});

test('direct execution patch application and both merge operands reject unknown fields', () => {
  const wrong = { setKernels: [] };
  assert.throws(() => applyExecutionPatch({}, wrong), /setKernels is unknown/);
  assert.throws(() => mergeExecutionPatchLists(wrong, {}), /setKernels is unknown/);
  assert.throws(() => mergeExecutionPatchLists({}, wrong), /setKernels is unknown/);
  for (const wrongType of [[], false, 0, 'set']) {
    assert.throws(() => mergeExecutionPatchLists({}, wrongType), /must be an object/);
  }
});

test('valid partial patch overlays preserve list replacement and nullish semantics', () => {
  const baseSet = [{ section: 'decode', op: 'q_proj', kernelKey: 'control' }];
  const kernels = [{ key: 'candidate', kernel: { kernel: 'test.wgsl', entry: 'main' } }];
  const merged = mergeExecutionPatchLists({ set: baseSet }, { addKernels: kernels });
  assert.strictEqual(merged.set, baseSet); assert.strictEqual(merged.addKernels, kernels);
  assert.deepEqual(mergeExecutionPatchLists({ set: baseSet }, { set: [] }).set, []);
  assert.strictEqual(mergeExecutionPatchLists({ set: baseSet }, { set: null }).set, baseSet);
  const graph = {};
  assert.strictEqual(applyExecutionPatch(graph, null), graph);
  assert.strictEqual(applyExecutionPatch(graph, {}), graph);
});
