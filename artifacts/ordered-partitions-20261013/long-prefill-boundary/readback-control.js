// Observation-only control: original activation bytes, no model computation.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const config = JSON.parse(await readFile(process.argv[2]));
const app = config.applicationDirectory;
const deliveryBytes = await readFile(app + '/native-delivery.json');
assert.equal(sha(deliveryBytes), config.nativeDeliverySha256);
const delivery = JSON.parse(deliveryBytes);
for (const entry of delivery.entries) {
  assert(!entry.path.startsWith('/') && !entry.path.split('/').includes('..'));
  assert.equal(sha(await readFile(app + '/' + entry.path)), entry.sha256);
}
const requireInstalled = createRequire(app + '/package.json');
const runtime = app + '/node_modules/doppler-gpu/src/';
const { verifyNumericalTestDelivery } = await import(pathToFileURL(
  '/var/tmp/reploid-recurrent-memory-20261013/cold-reference/numerical-test-delivery.js'));
const verifiedRuntime = await verifyNumericalTestDelivery({ ...config.testDelivery,
  runtimeDirectory: app + '/node_modules/doppler-gpu' });
const providerPath = requireInstalled.resolve('webgpu');
const { createNativeGpuHost } = await import(pathToFileURL(app + '/server/native-participant.js'));
const { initDevice, getDevice } = await import(pathToFileURL(runtime + 'gpu/device.js'));
const { snapshotTensor } = await import(pathToFileURL(runtime + 'debug/tensor.js'));
const { setGPUDevice } = await import(pathToFileURL(runtime + 'debug/config.js'));
const host = createNativeGpuHost(() => import(pathToFileURL(providerPath)));
const inputBytes = await readFile(process.argv[3]);
const input = JSON.parse(inputBytes)[0].activationInput;
const bytes = Buffer.from(input.data, 'base64');
assert.equal(bytes.byteLength, 184 * 1024 * 8);
assert.equal('sha256:' + sha(bytes), input.storageHash);
const receipt = { schema: 'doppler.immutable-buffer-readback-control/v1',
  hypothesis: 'Readback and staging retirement change an otherwise immutable source buffer',
  scope: 'Native Vulkan copy/readback only; no model, numerical qualification or provider substitution',
  inputSha256: sha(inputBytes), storageHash: input.storageHash, verifiedRuntime,
  providerEntrySha256: sha(await readFile(providerPath)), observations: [], closed: false };
let buffer;
try {
  await host.open();
  await initDevice();
  const device = getDevice();
  setGPUDevice(device);
  receipt.device = { vendor: device.adapterInfo?.vendor, description: device.adapterInfo?.description };
  device.addEventListener('uncapturederror', event => { (receipt.errors ??= []).push(event.error.message); });
  buffer = device.createBuffer({ label: 'immutable_original_activation', size: bytes.byteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  device.queue.writeBuffer(buffer, 0, bytes);
  for (let index = 0; index < 8; index++) {
    const capture = await snapshotTensor(buffer, [184, 1024], 'f32', {
      precision: 'two-f32', components: 2, elementStrideBytes: 8, includeData: false });
    assert(capture.ok, capture.error);
    receipt.observations.push({ index, storageDigest: capture.storageDigest,
      equal: capture.storageDigest === input.storageHash });
  }
  await device.queue.onSubmittedWorkDone();
  receipt.queueSettled = true;
  receipt.passed = receipt.observations.every(row => row.equal) && !receipt.errors?.length;
} catch (error) {
  receipt.failure = error.stack;
  process.exitCode = 1;
} finally {
  buffer?.destroy();
  try { await host.close(); receipt.closed = true; }
  catch (error) { receipt.cleanupFailure = error.stack; process.exitCode = 1; }
  setGPUDevice(null);
  await writeFile(process.argv[4], JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
}
if (!receipt.passed) process.exitCode = 1;
