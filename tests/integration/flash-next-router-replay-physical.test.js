import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { resolveDeclaredTopK } from '../../src/inference/pipelines/text/ffn/qwen-shared-expert.js';
import { runSoftmaxTopK } from '../../src/gpu/kernels/softmax.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const runPath = process.env.DOPPLER_FLASH_NEXT_ROUTER_REPLAY;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_ROUTER_REPLAY.');
  console.log('Flash-Next router selection replay: skipped (no provisioned observation)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath), run = JSON.parse(runBytes);
  assert.equal(typeof run.normalize, 'boolean');
  const observationBytes = await fs.readFile(run.observationPath), observation = JSON.parse(observationBytes);
  assert(observation.completed && !observation.routingControl && !observation.failure);
  assert.deepEqual(observation.gpuErrors, []);
  assert.deepEqual(observation.cleanupFailures, []);
  const modelRunBytes = await fs.readFile(run.modelRunPath), modelRun = JSON.parse(modelRunBytes);
  assert.equal(hash(modelRunBytes), observation.runSha256);
  const manifestBytes = await fs.readFile(path.join(modelRun.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), observation.manifestSha256);
  const manifest = JSON.parse(manifestBytes);
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason);
  const gpuErrors = [];
  getDevice().addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  const report = { scope: 'Physical declared GPU top-k replay of frozen logits; no tensor feedback or model qualification',
    normalModelQualified: false, process: { pid: process.pid, node: process.versions.node }, provider: probe.providerReceipt,
    observationSha256: hash(observationBytes), modelRunSha256: hash(modelRunBytes), rows: [], gpuErrors };
  let primaryError;
  try {
    for (const record of observation.rows[0].routerObservations) {
      const variant = resolveDeclaredTopK({ kernelPath: modelRun.runtimeConfig.inference.kernelPath, phase: 'prefill', layerIdx: record.layer });
      const raw = await fs.readFile(record.logits.file);
      assert.equal(hash(raw), record.logits.sha256);
      const [tokens, experts] = record.logits.shape;
      assert.equal(experts, manifest.moeConfig.numExperts);
      const topK = manifest.moeConfig.numExpertsPerToken;
      const buffers = [];
      const keep = buffer => { buffers.push(buffer); return buffer; };
      let operationError;
      try {
        const input = keep(acquireBuffer(raw.byteLength));
        uploadData(input, raw);
        const result = await runSoftmaxTopK(input, tokens, experts, topK, {
          variant, normalize: run.normalize, inputDtype: 'f32', weightsDtype: 'f32',
        });
        keep(result.indices); keep(result.weights);
        const bytes = Buffer.from(await readBuffer(result.indices, tokens * topK * 4));
        const file = `${run.outputPath}.layer${record.layer}.u32`;
        await fs.writeFile(file, bytes);
        report.rows.push({ layer: record.layer, variant, shape: [tokens, topK], file,
          sha256: hash(bytes), sourceLogitsSha256: record.logits.sha256 });
      } catch (error) {
        operationError = error;
        throw error;
      } finally {
        const failures = [];
        try { await getDevice().queue.onSubmittedWorkDone(); } catch (error) { failures.push(error); }
        for (const buffer of buffers) {
          try { releaseBuffer(buffer); } catch (error) { failures.push(error); }
        }
        if (failures.length) throw new AggregateError(operationError ? [operationError, ...failures] : failures, 'Router replay buffer cleanup failed', { cause: operationError });
      }
    }
    assert.deepEqual(gpuErrors, []);
    report.completed = true;
  } catch (error) {
    primaryError = error;
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    throw error;
  } finally {
    const failures = [];
    try { destroyDevice(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU(); } catch (error) { failures.push(error); }
    report.cleanupFailures = failures.map(String);
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(primaryError ? [primaryError, ...failures] : failures, 'Router replay cleanup failed', { cause: primaryError });
  }
}
