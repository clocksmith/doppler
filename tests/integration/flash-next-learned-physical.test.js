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
    inputsSha256: hash(inputBytes), runSha256: hash(runBytes), runtimeRevision: run.runtimeRevision,
    runtimeSourceSha256,
    modelId: manifest.modelId, quantizationInfo: manifest.quantizationInfo,
    provider: probe.providerReceipt, runtimeConfig: run.runtimeConfig, rows: [], gpuErrors };
  let pipeline, collectLayers;
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
        assert.ok(input.tokens.length <= 64, 'full layer captures are bounded to short diagnostic inputs');
        diagnostics = { enabled: true,
          captureConfig: { ...createDefaultCaptureConfig(), enabled: true, targetLevel: 'full',
            targetOpIds: Array.from({ length: manifest.architecture.numLayers }, (_, layer) => buildOpId('layer.out', layer)) } };
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
      const row = { id, inputTokenCount: input.tokens.length, tokenId: actualId, tokenIds: Array.from(output.tokenIds), logitRows,
        text: output.text ?? pipeline.tokenizer.decode(output.tokenIds),
        referenceTokenId: expected.argmax[position], referenceTop20Rank: referenceIds.indexOf(actualId) + 1,
        top1Agrees: actualId === expected.argmax[position],
        referenceTop20LogitDeltas: referenceIds.map((token, index) => ({ token,
          reference: referenceLogits[index], observed: observed[token], delta: observed[token] - referenceLogits[index] })) };
      if (run.captureLayers === true) {
        row.layers = collectLayers();
        assert.equal(captures.size, manifest.architecture.numLayers, 'every layer must be captured');
      }
      collectLayers = null;
      report.rows.push(row);
      await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
      console.log(JSON.stringify({ id, tokenId: actualId, text: row.text, top1Agrees: row.top1Agrees }));
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
    const failures = [];
    try { await pipeline?.unload(); } catch (error) { failures.push(error); }
    try { destroyDevice(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU(); } catch (error) { failures.push(error); }
    report.cleanupFailures = failures.map(error => String(error));
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(failures, 'Learned diagnostic cleanup failed');
  }
}
