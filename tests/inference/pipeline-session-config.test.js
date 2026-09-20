import assert from 'node:assert/strict';
import { getRuntimeConfig, setRuntimeConfig, snapshotRuntimeConfig } from '../../src/config/runtime.js';
import { scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';

const original = getRuntimeConfig();
function pipeline(temperature) {
  return scopePipelineShaders({
    runtimeConfig: snapshotRuntimeConfig({ inference: { sampling: { temperature } } }),
    async read() {
      await Promise.resolve();
      return getRuntimeConfig().inference.sampling.temperature;
    },
    async *stream() {
      yield getRuntimeConfig().inference.sampling.temperature;
      await Promise.resolve();
      yield getRuntimeConfig().inference.sampling.temperature;
    },
  }, null, { read: 'execution', stream: 'streaming' });
}
try {
  const a = pipeline(0.25), b = pipeline(0.75);
  setRuntimeConfig({ inference: { sampling: { temperature: 1.5 } } });
  assert.deepEqual(await Promise.all([a.read(), b.read()]), [0.25, 0.75]);
  assert.equal(getRuntimeConfig().inference.sampling.temperature, 1.5);
  assert.throws(() => { a.runtimeConfig.inference.sampling.temperature = 2; }, TypeError);
  const iterator = a.stream();
  assert.equal((await iterator.next()).value, 0.25);
  setRuntimeConfig({ inference: { sampling: { temperature: 1.75 } } });
  assert.equal(getRuntimeConfig().inference.sampling.temperature, 0.25, 'changing defaults cannot change the active request');
  const pendingB = b.read();
  await iterator.return();
  assert.equal(await pendingB, 0.75);
  assert.equal(getRuntimeConfig().inference.sampling.temperature, 1.75);
} finally {
  setRuntimeConfig(original);
}
