import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createImmediateResourceScope,
  createRecordedResourceScope,
} from '../../src/inference/resource-scope.js';

test('immediate failure cleanup releases every acquired stage exactly once', () => {
  for (let failAfter = 1; failAfter <= 5; failAfter += 1) {
    const released = [];
    const scope = createImmediateResourceScope({
      release: (resource) => released.push(resource.id),
    });
    const acquired = [];
    for (let stage = 1; stage <= failAfter; stage += 1) {
      const resource = { id: `stage-${stage}` };
      acquired.push(resource);
      scope.register(resource, resource.id, 'scopeOwned');
    }
    scope.close('failure');
    assert.deepEqual(released, acquired.map((resource) => resource.id));
  }
});

test('aliasing, transfer, retention, and output replacement are explicit', () => {
  const released = [];
  const scope = createImmediateResourceScope({
    release: (resource) => released.push(resource.id),
  });
  const sharedKV = { id: 'shared-kv' };
  const fusedNorm = { id: 'fused-norm' };
  const oldOutput = { id: 'old-output' };
  const finalOutput = { id: 'final-output' };

  scope.register(sharedKV, 'key', 'scopeOwned');
  scope.register(sharedKV, 'value-alias', 'scopeOwned');
  scope.transfer(sharedKV, 'transferred', 'shared-kv-store');
  scope.register(fusedNorm, 'fused-norm', 'scopeOwned');
  scope.register(oldOutput, 'old-output', 'scopeOwned');
  scope.register(finalOutput, 'final-output', 'scopeOwned');
  scope.release(oldOutput);
  scope.retain(finalOutput, 'final-output', 'returned-to-caller');
  const events = scope.close('success');

  assert.deepEqual(released.sort(), ['fused-norm', 'old-output']);
  assert.equal(events.filter((event) => event.action === 'alias').length, 1);
  assert.equal(events.some((event) => event.detail === 'shared-kv-store'), true);
  assert.equal(events.some((event) => event.detail === 'returned-to-caller'), true);
});

test('recorded scope retains submit-owned buffers and prevents duplicate tracking', () => {
  const tracked = [];
  const recorder = {
    trackTemporaryBuffer(resource) {
      tracked.push(resource.id);
    },
  };
  const scope = createRecordedResourceScope(recorder);
  const keyAndValue = { id: 'aliased-kv' };
  const output = { id: 'output' };

  scope.register(keyAndValue, 'key', 'submitOwned');
  scope.register(keyAndValue, 'value', 'submitOwned');
  scope.release(keyAndValue);
  scope.release(keyAndValue);
  scope.register(output, 'output', 'scopeOwned');
  scope.retain(output, 'output', 'returned-to-caller');
  const events = scope.close('success');

  assert.deepEqual(tracked, ['aliased-kv']);
  assert.equal(events.filter((event) => event.action === 'submit-retain').length, 1);
  assert.equal(events.filter((event) => event.action === 'release-skip').length, 1);
});

test('early return closes owned temporaries while preserving borrowed resources', () => {
  const released = [];
  const scope = createImmediateResourceScope({
    release: (resource) => released.push(resource.id),
  });
  scope.register({ id: 'borrowed' }, 'borrowed', 'borrowed');
  scope.register({ id: 'temporary' }, 'temporary', 'scopeOwned');
  scope.close('success');
  assert.deepEqual(released, ['temporary']);
});

test('pooled resource reacquisition starts a new ownership lifetime', () => {
  const pooledBuffer = { id: 'pooled-buffer' };
  const released = [];
  const scope = createImmediateResourceScope({
    release: (resource) => released.push(resource.id),
  });

  scope.register(pooledBuffer, 'first-use', 'scopeOwned');
  scope.release(pooledBuffer);
  scope.register(pooledBuffer, 'reacquired-output', 'scopeOwned');
  scope.retain(pooledBuffer, 'reacquired-output', 'returned-to-caller');
  const events = scope.close('success');

  assert.deepEqual(released, ['pooled-buffer']);
  assert.equal(events.some((event) => (
    event.action === 'reacquire'
    && event.label === 'reacquired-output'
    && event.detail === 'first-use'
  )), true);
  assert.equal(events.some((event) => event.detail === 'returned-to-caller'), true);
});
