import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { installHyperConnectionObservation } from '../helpers/hyper-connection-observation-loader.js';

const runPath = process.env.DOPPLER_FLASH_NEXT_UPSTREAM_TRACE;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_UPSTREAM_TRACE.');
  console.log('Flash-Next upstream trace: skipped (no provisioned run)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath), run = JSON.parse(runBytes);
  const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), run.expectedManifestSha256);
  const manifest = JSON.parse(manifestBytes);
  const inputsBytes = await fs.readFile(run.inputsPath), inputs = JSON.parse(inputsBytes);
  const input = inputs.sequences.find(row => row.id === 'long');
  assert(input && input.tokens.length === 2304);
  const sourceURL = new URL('../../src/inference/pipelines/text/hyper-connection/decoder-layer.js', import.meta.url);
  const originalSource = await fs.readFile(sourceURL);
  const report = { scope: 'Layer-0 upstream observation with intentional abort before MoE; not a generated answer or model qualification',
    normalModelQualified: false, process: { pid: process.pid, node: process.versions.node },
    manifestSha256: hash(manifestBytes), inputsSha256: hash(inputsBytes), runSha256: hash(runBytes), tensors: [] };
  const controller = new AbortController(), stop = new Error('trace: requested upstream boundary captured');
  const stopLayer = run.stopBeforeMoeLayer ?? 0;
  assert(Number.isSafeInteger(stopLayer) && stopLayer >= 0 && stopLayer < manifest.architecture.numLayers);
  const traceLayers = run.traceLayers ?? [stopLayer];
  assert(traceLayers.includes(stopLayer));
  report.scope = `Layer-${stopLayer} upstream observation with intentional abort before MoE; not a generated answer or model qualification`;
  let layer = -1, readBuffer, readBufferSlice;
  const observer = installHyperConnectionObservation(sourceURL.href, hash(originalSource), async record => {
    if (record.role === 'attn_hc' && record.stage === 'input') layer++;
    if (!traceLayers.includes(layer) || record.role === 'hc_mixer') return;
    assert.equal(record.tensor.dtype, 'f32');
    let bytes, shape = record.tensor.shape;
    if (run.positions) {
      assert.equal(shape.length, 2);
      assert(run.positions.every(position => Number.isSafeInteger(position) && position >= 0 && position < shape[0]));
      const rows = [];
      for (const position of run.positions) rows.push(Buffer.from(await readBufferSlice(record.tensor.buffer, position * shape[1] * 4, shape[1] * 4)));
      bytes = Buffer.concat(rows);
      shape = [run.positions.length, shape[1]];
    } else {
      bytes = Buffer.from(await readBuffer(record.tensor.buffer, shape.reduce((a, b) => a * b, 1) * 4));
    }
    const file = `${run.outputPath}.L${layer}.${record.role}.${record.stage}.f32`;
    await fs.writeFile(file, bytes);
    report.tensors.push({ layer, role: record.role, stage: record.stage, shape, originalShape: record.tensor.shape,
      positions: run.positions ?? null, file, sha256: hash(bytes) });
    if (layer === stopLayer && record.role === 'mlp_hc' && record.stage === 'branchInput') controller.abort(stop);
  });
  let pipeline, destroyDevice, releaseNodeWebGPU, qsaObservation;
  const gpuErrors = [];
  let primaryError;
  try {
    const [gpu, runtime, storage, memory, device, provider, capture, identity, stages, qsa] = await Promise.all([
      import('../helpers/gpu-probe.js'), import('../../src/inference/pipelines/text.js'),
      import('../../src/storage/artifact-storage-context.js'), import('../../src/memory/buffer-pool.js'),
      import('../../src/gpu/device.js'), import('../../src/tooling/node-webgpu.js'),
      import('../../src/debug/capture-policy.js'), import('../../src/inference/pipelines/text/operator-identity.js'),
      import('../../src/inference/pipelines/text/stage-names.js'),
      import('../helpers/learned-qsa-observation.js'),
    ]);
    readBuffer = memory.readBuffer;
    readBufferSlice = memory.readBufferSlice;
    destroyDevice = device.destroyDevice;
    releaseNodeWebGPU = provider.releaseNodeWebGPU;
    const probe = await gpu.probeNodeGPU({ installFileFetchShim: true });
    assert(probe.ready, probe.reason);
    report.provider = probe.providerReceipt;
    device.getDevice().addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
    pipeline = await runtime.createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
      storage: storage.createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
    if (run.captureQsaLayers) {
      const reference = JSON.parse(await fs.readFile(run.referencePath));
      qsaObservation = qsa.observeLearnedQsa(run.captureQsaLayers, reference.sequences.long.qsa,
        run.outputPath, run.captureQsaPositions ?? []);
    }
    if (run.auditWeights === true) {
      report.weights = [];
      const groups = pipeline.weights.get(`layer_${stopLayer}`).hyperConnections;
      for (const [role, group, prefix] of [['attn_hc', groups.attention, 'attn_hyper_connection'], ['mlp_hc', groups.mlp, 'mlp_hyper_connection']]) {
        for (const [field, suffix] of [['norm', 'hc_norm'], ['down', 'input_mix_weight_down'], ['up', 'input_mix_weight_up'], ['inject', 'block_inject_weight']]) {
          const name = `model.language_model.layers.${stopLayer}.${prefix}.${suffix}.weight`;
          const descriptor = manifest.tensors[name];
          assert(descriptor && descriptor.dtype === 'F16');
          const weight = group[field];
          if (field !== 'norm') assert.equal(weight.dtype, 'f32');
          const bytes = Buffer.from(await readBuffer(weight.buffer ?? weight, descriptor.shape.reduce((a, b) => a * b, 1) * 4));
          const file = `${run.outputPath}.${role}.weight-${field}.f32`;
          await fs.writeFile(file, bytes);
          report.weights.push({ tensor: name, shape: descriptor.shape, file, sha256: hash(bytes) });
        }
      }
    }
    const targetStages = run.captureLinearStages === false ? [] : ['post_input_norm', 'linear_qkv_proj', 'linear_z_proj', 'linear_a_proj', 'linear_b_proj', 'linear_core_out'];
    await assert.rejects(pipeline.generateTokenIds('', { inputIds: input.tokens, useChatTemplate: false,
      signal: controller.signal, maxTokens: 1, temperature: 0, topK: 1, topP: 1, seed: 0,
      diagnostics: targetStages.length ? { enabled: true, captureConfig: { ...capture.createDefaultCaptureConfig(), enabled: true,
        targetLevel: 'full', targetOpIds: targetStages.map(stage => identity.buildOpId(stages.PROBE_TO_CANONICAL[stage], stopLayer)) } } : undefined,
    }), error => error === stop);
    for (const record of pipeline.stats.operatorDiagnostics?.timeline ?? []) {
      if (!record.capture?.data || record.layerIndex !== stopLayer) continue;
      const bytes = Buffer.from(Float32Array.from(record.capture.data).buffer);
      const file = `${run.outputPath}.${record.stageName}.f32`;
      await fs.writeFile(file, bytes);
      report.tensors.push({ layer: stopLayer, role: 'linear', stage: record.stageName, shape: record.capture.shape, file, sha256: hash(bytes) });
    }
    assert(report.tensors.some(row => row.layer === stopLayer && row.role === 'mlp_hc' && row.stage === 'branchInput'));
    assert.equal(report.tensors.filter(row => row.role === 'linear').length, targetStages.length);
    if (qsaObservation) {
      assert.deepEqual(qsaObservation.errors, []);
      report.qsaObservations = qsaObservation.records;
    }
    assert.deepEqual(gpuErrors, []);
    report.diagnosticCompleted = true;
  } catch (error) {
    primaryError = error;
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    throw error;
  } finally {
    const failures = [];
    try { await pipeline?.unload(); } catch (error) { failures.push(error); }
    try { destroyDevice?.(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU?.(); } catch (error) { failures.push(error); }
    observer.restore();
    qsaObservation?.restore();
    report.instrumentation = observer.receipt;
    report.gpuErrors = gpuErrors;
    report.cleanupFailures = failures.map(String);
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(primaryError ? [primaryError, ...failures] : failures, 'Upstream trace cleanup failed', { cause: primaryError });
  }
}
