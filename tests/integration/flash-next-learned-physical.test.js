import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { compareLearnedLayerCapture } from '../helpers/learned-layer-observation.js';
import { createDefaultCaptureConfig } from '../../src/debug/capture-policy.js';
import { buildOpId } from '../../src/inference/pipelines/text/operator-identity.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
import { checkLearnedLifecycle } from '../helpers/flash-next-lifecycle.js';
import { observeLearnedQsa } from '../helpers/learned-qsa-observation.js';
import { observeLearnedRouters } from '../helpers/learned-router-observation.js';

// Provisioned learned-weight diagnostic. This is not a performance test or a
// qualification threshold: quantization spread and intermediate parity need
// separate controls. The immutable upstream reference never reads Doppler output.
const runPath = process.env.DOPPLER_FLASH_NEXT_PROBE;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_PROBE to the explicit run JSON.');
  console.log('flash-next learned diagnostic: skipped (no provisioned artifact/reference)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath);
  const run = JSON.parse(runBytes);
  assert.equal(run.schema, 'doppler.flash-next-learned-probe/v1');
  assert.ok(run.runtimeConfig && run.modelDir && run.referencePath && run.inputsPath && run.outputPath);
  const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), run.expectedManifestSha256, 'the probe must name the exact prepared manifest');
  const manifest = JSON.parse(manifestBytes);
  const referenceBytes = await fs.readFile(run.referencePath);
  const reference = JSON.parse(referenceBytes);
  const inputBytes = await fs.readFile(run.inputsPath);
  const inputs = JSON.parse(inputBytes);
  assert.equal(reference.schema, 'doppler.qwen38-learned-forward-reference/v1');
  assert.equal(reference.checkpoint, manifest.artifactIdentity.sourceRevision);
  assert.equal(reference.dtype, 'float32');
  if (reference.artifactManifestSha256) assert.equal(reference.artifactManifestSha256, hash(manifestBytes));
  assert.equal(manifest.inference.hyperConnections.semantics, 'qwen4_exp_gated_residual');
  const runtimeSourceSha256 = {};
  for (const file of run.sourceFiles ?? []) runtimeSourceSha256[file] = hash(await fs.readFile(file));
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason);
  const gpuErrors = [];
  getDevice().addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  const report = { schema: 'doppler.flash-next-learned-observation/v1',
    scope: 'Node physical load and last-position logits diagnostic on the learned quantized artifact; no general qualification or performance claim',
    manifestSha256: hash(manifestBytes), referenceSha256: hash(referenceBytes),
    referenceScope: reference.scope,
    inputsSha256: hash(inputBytes), runSha256: hash(runBytes), runtimeRevision: run.runtimeRevision,
    runtimeSourceSha256,
    modelId: manifest.modelId, quantizationInfo: manifest.quantizationInfo,
    provider: probe.providerReceipt, runtimeConfig: run.runtimeConfig, rows: [], gpuErrors };
  let pipeline, collectLayers, qsaObservation, routerObservation;
  try {
    pipeline = await createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
      storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
    report.loaded = true;
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    for (const { id, maxTokens } of run.sequences) {
      assert.ok(Number.isSafeInteger(maxTokens) && maxTokens > 0);
      const input = inputs.sequences.find(sequence => sequence.id === id);
      const expected = reference.sequences[id];
      assert.ok(input && expected);
      assert.equal(input.tokens.length, expected.tokens);
      pipeline.resetGenerationState();
      qsaObservation = run.captureQsaLayers
        ? observeLearnedQsa(run.captureQsaLayers, expected.qsa, `${run.outputPath}.${id}`) : null;
      routerObservation = run.captureRouterLayers
        ? observeLearnedRouters(run.captureRouterLayers, `${run.outputPath}.${id}`) : null;
      const captures = new Map();
      collectLayers = () => {
        for (const record of pipeline.stats.operatorDiagnostics?.timeline ?? []) {
          if (record.stageName === 'layer.out' && record.capture?.shape?.[0] === input.tokens.length && !captures.has(record.layerIndex)) {
            captures.set(record.layerIndex, record.capture);
          }
        }
        return [...captures].map(([layer, capture]) => ({ layer,
          ...compareLearnedLayerCapture(capture, expected.layers[layer], expected.samplePositions, reference.sampleStride) }));
      };
      let diagnostics;
      if (run.captureLayers === true) {
        const captureLayerIds = run.captureLayerIds ?? Array.from({ length: manifest.architecture.numLayers }, (_, layer) => layer);
        assert.ok(captureLayerIds.length && new Set(captureLayerIds).size === captureLayerIds.length);
        assert.ok(captureLayerIds.every(layer => Number.isSafeInteger(layer) && layer >= 0 && layer < manifest.architecture.numLayers));
        assert.ok(input.tokens.length <= 64 || (input.tokens.length <= 4096 && captureLayerIds.length <= 4),
          'long-input full captures require at most four explicitly selected layers');
        diagnostics = { enabled: true,
          captureConfig: { ...createDefaultCaptureConfig(), enabled: true, targetLevel: 'full',
            targetOpIds: captureLayerIds.map(layer => buildOpId('layer.out', layer)) } };
      }
      let observed = null;
      let logitRows = 0;
      const output = await pipeline.generateTokenIds('', { inputIds: input.tokens, useChatTemplate: false,
        maxTokens, temperature: 0, topK: 1, topP: 1, seed: 0, repetitionPenalty: 1,
        diagnostics, onLogits: logits => {
          assert.ok(logits.every(Number.isFinite), 'every prefill/decode logit row must be finite');
          if (observed === null) observed = Array.from(logits);
          logitRows++;
        } });
      assert.ok(observed?.length === manifest.architecture.vocabSize, 'complete last-position logits must be observed');
      assert.ok(observed.every(Number.isFinite), 'learned logits must be finite');
      const position = input.tokens.length - 1;
      const referenceIds = expected.top20Ids[position];
      const referenceLogits = expected.top20Logits[position];
      const actualId = output.tokenIds[0];
      const prefillLogitsFile = `${run.outputPath}.${id}.prefill.f32`;
      const prefillLogitsBytes = Buffer.from(Float32Array.from(observed).buffer);
      await fs.writeFile(prefillLogitsFile, prefillLogitsBytes);
      const row = { id, inputTokenCount: input.tokens.length, tokenId: actualId, tokenIds: Array.from(output.tokenIds), logitRows,
        prefillLogitsFile, prefillLogitsSha256: hash(prefillLogitsBytes),
        bufferPool: getBufferPool().getStats(),
        residentExpertLayers: [...pipeline.dopplerLoader.experts.keys()].filter(key => /^layer_\d+_qwen35_packed$/.test(key))
          .map(key => Number(key.split('_')[1])),
        text: output.text ?? pipeline.tokenizer.decode(output.tokenIds),
        referenceTokenId: expected.argmax[position], referenceTop20Rank: referenceIds.indexOf(actualId) + 1,
        top1Agrees: actualId === expected.argmax[position],
        referenceTop20LogitDeltas: referenceIds.map((token, index) => ({ token,
          reference: referenceLogits[index], observed: observed[token], delta: observed[token] - referenceLogits[index] })) };
      if (run.captureLayers === true) {
        row.layers = collectLayers();
        assert.equal(captures.size, run.captureLayerIds?.length ?? manifest.architecture.numLayers, 'every requested layer must be captured');
        if (run.retainLayerFiles === true) for (const layer of row.layers) {
          const bytes = Buffer.from(Float32Array.from(captures.get(layer.layer).data).buffer);
          layer.file = `${run.outputPath}.${id}.layer${layer.layer}.f32`;
          layer.sha256 = hash(bytes);
          await fs.writeFile(layer.file, bytes);
        }
      }
      const pager = run.runtimeConfig.loading?.expertCache?.qwen35Pager;
      if (pager?.enabled) assert.ok(row.residentExpertLayers.length <= pager.maxResidentLayers, 'resident expert banks must remain bounded');
      if (qsaObservation) {
        row.qsaObservations = qsaObservation.records;
        assert.deepEqual(qsaObservation.errors, []);
        qsaObservation.restore(); qsaObservation = null;
      }
      if (routerObservation) {
        row.routerObservations = routerObservation.records;
        assert.deepEqual(routerObservation.errors, []);
        routerObservation.restore(); routerObservation = null;
      }
      collectLayers = null;
      report.rows.push(row);
      await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
      console.log(JSON.stringify({ id, tokenId: actualId, text: row.text, top1Agrees: row.top1Agrees }));
    }
    if (run.lifecycle === true) {
      const baseline = report.rows[0];
      const input = inputs.sequences.find(sequence => sequence.id === baseline.id);
      report.lifecycle = [];
      await checkLearnedLifecycle(pipeline, input, baseline.tokenIds, async result => {
        report.lifecycle.push(result);
        await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
        console.log(JSON.stringify(result));
      });
      pipeline = await createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
        storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
      const fresh = await pipeline.generateTokenIds('', { inputIds: input.tokens, useChatTemplate: false,
        maxTokens: 1, temperature: 0, topK: 1, topP: 1, seed: 0, repetitionPenalty: 1 });
      assert.equal(fresh.tokenIds[0], baseline.tokenIds[0]);
      report.lifecycle.push({ case: 'reload-and-reuse', passed: true, tokenIds: Array.from(fresh.tokenIds) });
    }
    await getDevice().queue.onSubmittedWorkDone();
    assert.deepEqual(gpuErrors, []);
    report.completed = true;
  } catch (error) {
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    try { if (collectLayers) report.partialLayers = collectLayers(); }
    catch (captureError) { report.captureFailure = String(captureError); }
    throw error;
  } finally {
    qsaObservation?.restore();
    routerObservation?.restore();
    const failures = [];
    try { await pipeline?.unload(); } catch (error) { failures.push(error); }
    try { destroyDevice(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU(); } catch (error) { failures.push(error); }
    report.cleanupFailures = failures.map(error => String(error));
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(failures, 'Learned diagnostic cleanup failed');
  }
}
