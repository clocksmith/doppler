import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { load } from 'doppler-gpu/compat';
import { applyRuntimeProfile } from 'doppler-gpu/tooling/runtime';
import { getDevice, destroyDevice } from 'doppler-gpu/tooling/device';

// Copy this provisioned consumer outside the source tree, beside an extracted
// npm package. Runtime imports intentionally use only published exports.
const runPath = process.env.DOPPLER_FLASH_NEXT_CONSUMER;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_CONSUMER.');
  console.log('Flash-Next public consumer: skipped (no explicit package/artifact run)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath), run = JSON.parse(runBytes);
  assert.equal(run.schema, 'doppler.flash-next-public-consumer-run/v1');
  const modelBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(modelBytes), run.expectedManifestSha256);
  const referenceBytes = await fs.readFile(run.referencePath), reference = JSON.parse(referenceBytes);
  assert.equal(reference.artifactManifestSha256, hash(modelBytes));
  assert.equal(reference.kvStorageDtype, 'f16');
  const inputsBytes = await fs.readFile(run.inputsPath), inputs = JSON.parse(inputsBytes);
  const packageReceipt = JSON.parse(await fs.readFile(run.packageReceiptPath));
  assert.equal(packageReceipt.packageSha256, run.expectedPackageSha256);
  const runtimeRoot = fileURLToPath(new URL('./config/runtime/', import.meta.resolve('doppler-gpu/compat')));
  const server = http.createServer(async (request, response) => {
    try {
      const name = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const file = path.resolve(runtimeRoot, `.${name}`);
      if (!file.startsWith(runtimeRoot) || !file.endsWith('.json')) throw Error('Not found');
      response.setHeader('Content-Type', 'application/json');
      response.end(await fs.readFile(file));
    } catch { response.statusCode = 404; response.end(); }
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const report = { schema: 'doppler.flash-next-public-consumer/v1',
    scope: 'Installed compatibility API, named runtime profile and receipted generation; no participant service changes',
    runSha256: hash(runBytes), consumerSha256: hash(await fs.readFile(fileURLToPath(import.meta.url))),
    manifestSha256: hash(modelBytes), referenceSha256: hash(referenceBytes), inputsSha256: hash(inputsBytes),
    package: { sha256: packageReceipt.packageSha256, dependencySource: packageReceipt.dependencySource },
    rows: [], gpuErrors: [] };
  let model;
  try {
    const runtimeConfig = await applyRuntimeProfile(run.runtimeProfile,
      { baseUrl: `http://127.0.0.1:${server.address().port}` });
    model = await load({ url: pathToFileURL(run.modelDir).href }, { runtimeConfig, isolatedLoader: true });
    assert.equal(model.manifestHash, hash(modelBytes));
    assert.equal(model.resolvedArtifactVariantId, `sha256:${hash(modelBytes)}`);
    assert.equal(model.manifest.moeConfig.expertFormat, 'qwen3_5');
    assert.ok(getDevice(), 'public device export must identify the loaded host device');
    getDevice().addEventListener('uncapturederror', event => report.gpuErrors.push(event.error.message));
    report.loaded = true;
    report.artifactIdentity = model.manifest.artifactIdentity;
    for (const id of run.sequences) {
      const input = inputs.sequences.find(row => row.id === id);
      model.resetGenerationState();
      let observed;
      const evidence = await model.generateWithEvidence(input.text, { inputIds: input.tokens,
        useChatTemplate: false, maxTokens: 1, seed: 0,
        onLogits(values) { assert(values.every(Number.isFinite)); observed = Float32Array.from(values); } });
      assert.equal(evidence.schema, 'doppler_generation_evidence/v1');
      assert.deepEqual(evidence.tokenIds, [reference.sequences[id].argmax[input.tokens.length - 1]]);
      assert.ok(evidence.resolution.resolvedExecutionId);
      assert.equal(evidence.resolution.resolvedArtifactVariantId, model.resolvedArtifactVariantId);
      const prefillLogitsFile = `${run.outputPath}.${id}.prefill.f32`;
      const bytes = Buffer.from(observed.buffer); await fs.writeFile(prefillLogitsFile, bytes);
      report.rows.push({ id, inputTokenCount: input.tokens.length, tokenId: evidence.tokenIds[0], tokenIds: evidence.tokenIds,
        prefillLogitsFile, prefillLogitsSha256: hash(bytes), evidence });
      await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
      console.log(JSON.stringify({ id, tokenIds: evidence.tokenIds, text: evidence.outputText }));
    }
    assert.deepEqual(report.gpuErrors, []);
    report.completed = true;
  } catch (error) {
    report.failure = { name: error.name, message: error.message, stack: error.stack }; throw error;
  } finally {
    const failures = [];
    try { await model?.unload(); } catch (error) { failures.push(String(error)); }
    try { destroyDevice(); } catch (error) { failures.push(String(error)); }
    server.closeAllConnections(); await new Promise(resolve => server.close(resolve));
    report.cleanupFailures = failures;
    await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
    assert.deepEqual(failures, []);
  }
}
