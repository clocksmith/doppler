import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import * as runtime from '../../src/capsule-runtime.js';
import * as partitions from 'doppler-gpu/partitions';
import { createForecastProgramFactory } from '../../src/client/runtime/composition-root.js';

assert.equal(typeof runtime.openCapsule, 'function');
assert.equal('createLayerPartitionPlan' in runtime, false);
for (const name of ['createLayerPartitionPlan', 'serializeActivationFrame',
  'deserializeActivationFrame', 'comparePartitionExecution']) {
  assert.equal(typeof partitions[name], 'function', `Public partition contract missing: ${name}`);
}
const partition = partitions.createLayerPartitionPlan({ modelId: 'contract-test', numLayers: 4,
  hiddenSize: 2, vocabSize: 8, splitLayer: 2, activationDtype: 'f32' });
assert.deepEqual(partition.partitions.map(group => group.layerRange), [[0, 1], [2, 3]]);
const activation = partitions.serializeActivationFrame({ shape: [1, 1, 2], dtype: 'f32',
  data: new Float32Array([1, 2]), seqOffset: 3, step: 1, metadata: { attemptId: 'a' } });
assert.deepEqual([...partitions.deserializeActivationFrame(activation).tensorData], [1, 2]);
assert.equal(runtime.createDopplerRun, runtime.createDopplerRuntime);
assert.equal(runtime.RUN_CORE_VERSION, runtime.RUNTIME_CORE_VERSION);
assert.equal(await import('doppler-gpu/run'), runtime);
assert.equal(await import('doppler-gpu/runtime'), runtime);
assert.equal('openPack' in runtime, false);
assert.equal('createFetchPackArtifactStore' in runtime, false);
assert.equal(typeof runtime.createDopplerRuntime, 'function');
assert.equal(typeof runtime.createFetchCapsuleArtifactStore, 'function');
assert.equal(typeof runtime.createForecastProgramFactory, 'function');
assert.equal(runtime.createForecastProgramFactory, createForecastProgramFactory);
assert.throws(() => runtime.createForecastProgramFactory(null), /explicit GPUDevice/);
assert.equal('load' in runtime, false);
assert.equal('open' in runtime, false);
assert.equal('createDopplerProvider' in runtime, false);
assert.throws(() => runtime.openCapsule({}, {}), /explicit ports/);

const packageJson = JSON.parse(await fs.readFile('package.json', 'utf8'));
assert.equal(packageJson.main, 'src/capsule-runtime.js');
assert.equal(packageJson.types, 'src/capsule-runtime.d.ts');
assert.equal(packageJson.exports['.'].import, './src/capsule-runtime.js');
assert.equal(packageJson.exports['.'].types, './src/capsule-runtime.d.ts');
assert.equal(packageJson.exports['./runtime'].import, './src/capsule-runtime.js');
assert.equal(packageJson.exports['./capsule'].import, './src/capsule.js');
assert.equal(Object.hasOwn(packageJson.exports, './pack'), false);
await assert.rejects(import('doppler-gpu/pack'), { code: 'ERR_PACKAGE_PATH_NOT_EXPORTED' });
for (const file of ['src/pack-runtime.js', 'src/pack.js', 'src/client/pack-host.js']) {
  await assert.rejects(fs.access(file), { code: 'ENOENT' });
}
assert.equal(packageJson.exports['./compat'].import, './src/index.js');

console.log('✔ capsule-runtime-entrypoint.test.js passed');
