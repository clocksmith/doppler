import test from 'node:test';
import assert from 'node:assert/strict';
import { ObservedSession } from './observed_session.js';

function setup(location = 'gpu-buffer') {
  const disposed = [];
  const tensor = name => ({ location, type: 'float32', dims: [2], dispose() { disposed.push(name); } });
  const calls = [];
  const session = { inputNames: ['input'], inputMetadata: [], outputNames: ['logits', 'weight'],
    outputMetadata: [{ name: 'logits' }, { name: 'weight' }],
    async run(feeds, names) { calls.push(names); return Object.fromEntries(names.map(name => [name, tensor(name)])); } };
  const port = { output: 'weight', type: 'float32', shape: [2] };
  return { disposed, calls, session, port };
}

test('initializer observation uses first actual invocation and preserves model port ownership', async () => {
  const { disposed, calls, session, port } = setup();
  const observations = [];
  const facade = new ObservedSession(session, [port], async (descriptor, tensor) => observations.push({ descriptor, tensor }));
  assert.deepEqual(facade.outputNames, ['logits']);
  assert.deepEqual(facade.outputMetadata, [{ name: 'logits' }]);
  const first = await facade.run({ input: 'actual' });
  assert.deepEqual(Object.keys(first), ['logits']);
  assert.deepEqual(disposed, ['weight']);
  await facade.run({ input: 'next' });
  assert.equal(observations.length, 1);
  assert.deepEqual(calls, [['logits', 'weight'], ['logits']]);
});

test('observation rejects CPU weight outputs and releases every produced tensor on failure', async () => {
  const { disposed, session, port } = setup('cpu');
  const facade = new ObservedSession(session, [port], async () => assert.fail('must not observe CPU output'));
  await assert.rejects(facade.run({}), /actual declared F32 GPU buffer/);
  assert.deepEqual(disposed.sort(), ['logits', 'weight']);
});
