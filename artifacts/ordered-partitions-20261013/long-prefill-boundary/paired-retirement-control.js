// Isolated installed paired addition; no full-model qualification.
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
const { acquireBuffer, releaseBuffer, destroyBufferPool } = await import(pathToFileURL(runtime + 'memory/buffer-pool.js'));
const shaderBytes = await readFile(runtime + 'gpu/kernels/paired_add.wgsl');
const host = createNativeGpuHost(() => import(pathToFileURL(providerPath)));
const inputBytes = await readFile(process.argv[3]);
const input = JSON.parse(inputBytes)[0].activationInput;
const bytes = Buffer.from(input.data, 'base64');
assert.equal(bytes.byteLength, 184 * 1024 * 8);
assert.equal('sha256:' + sha(bytes), input.storageHash);
const receipt = { schema: 'doppler.paired-compute-retirement-control/v1',
  hypothesis: 'Temporary-buffer retirement changes a completed paired computation output',
  scope: 'Installed paired-add shader and pool retirement; no full-model qualification or provider substitution',
  shaderSha256: sha(shaderBytes),
  inputSha256: sha(inputBytes), storageHash: input.storageHash, verifiedRuntime,
  providerEntrySha256: sha(await readFile(providerPath)), observations: [], closed: false };
let buffer, output, second, uniform;
const temporary = [];
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
  const inputValues = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const addend = new Float32Array(inputValues.length);
  for (let i = 0; i < addend.length; i += 2) addend[i] = 0.125;
  second = device.createBuffer({ label: 'paired_addend', size: bytes.byteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
  device.queue.writeBuffer(second, 0, addend);
  uniform = device.createBuffer({ label: 'paired_uniform', size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uniforms = new ArrayBuffer(16);
  new Uint32Array(uniforms)[0] = 184 * 1024;
  new Float32Array(uniforms)[1] = 1;
  device.queue.writeBuffer(uniform, 0, uniforms);
  const module = device.createShaderModule({ code: shaderBytes.toString(), label: 'installed_paired_add' });
  const compilation = await module.getCompilationInfo();
  receipt.compilationMessages = compilation.messages.map(m => ({type:m.type, message:m.message}));
  assert(!compilation.messages.some(m => m.type === 'error'));
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto',
    compute: { module, entryPoint: 'main', constants: { WORKGROUP_SIZE: 256 } } });
  output = acquireBuffer(bytes.byteLength, undefined, 'diagnostic_paired_output');
  for (let i = 0; i < 4; i++) temporary.push(acquireBuffer(bytes.byteLength, undefined, 'diagnostic_temporary'));
  const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [uniform, buffer, second, output].map((value, binding) => ({binding,resource:{buffer:value}})) });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(Math.ceil(184 * 1024 / 256)); pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  for (let index = 0; index < 8; index++) {
    if (index === 4) {
      for (const value of temporary.splice(0)) releaseBuffer(value);
    }
    const capture = await snapshotTensor(output, [184, 1024], 'f32', {
      precision: 'two-f32', components: 2, elementStrideBytes: 8, includeData: true });
    assert(capture.ok, capture.error);
    let maxAbsoluteError = 0, zeroRows = 0;
    for (let row = 0; row < 184; row++) {
      let allZero = true;
      for (let column = 0; column < 1024; column++) {
        const i = (row * 1024 + column) * 2;
        const actual = capture.storageComponents[i] + capture.storageComponents[i + 1];
        const expected = inputValues[i] + inputValues[i + 1] + 0.125;
        maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual - expected));
        allZero &&= actual === 0;
      }
      if (allZero) zeroRows++;
    }
    receipt.observations.push({ index, retired: index >= 4, storageDigest: capture.storageDigest,
      maxAbsoluteError, zeroRows });
  }
  await device.queue.onSubmittedWorkDone();
  receipt.queueSettled = true;
  receipt.passed = receipt.observations.every(row => row.maxAbsoluteError < 2e-11
    && row.storageDigest === receipt.observations[0].storageDigest) && !receipt.errors?.length;
} catch (error) {
  receipt.failure = error.stack;
  process.exitCode = 1;
} finally {
  for (const value of temporary) releaseBuffer(value);
  if (output) releaseBuffer(output);
  destroyBufferPool();
  buffer?.destroy(); second?.destroy(); uniform?.destroy();
  try { await host.close(); receipt.closed = true; }
  catch (error) { receipt.cleanupFailure = error.stack; process.exitCode = 1; }
  setGPUDevice(null);
  await writeFile(process.argv[4], JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
}
if (!receipt.passed) process.exitCode = 1;
