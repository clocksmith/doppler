import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { load } from 'doppler-gpu/compat';
import { applyRuntimeProfile, applyRuntimeConfigFromUrl } from 'doppler-gpu/tooling/runtime';
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
  const expectedKvDtype = run.expectedKvDtype ?? 'f16';
  assert.ok(['f16', 'f32'].includes(expectedKvDtype));
  assert.equal(reference.kvStorageDtype ?? 'f32', expectedKvDtype);
  const inputsBytes = await fs.readFile(run.inputsPath), inputs = JSON.parse(inputsBytes);
  const packageReceipt = JSON.parse(await fs.readFile(run.packageReceiptPath));
  assert.equal(packageReceipt.packageSha256, run.expectedPackageSha256);
  if (run.packageTarballPath) {
    assert.equal(hash(await fs.readFile(run.packageTarballPath)), run.expectedPackageSha256);
    assert.ok(packageReceipt.files?.length, 'installed candidate needs a complete package-file inventory');
    const packageRoot = fileURLToPath(new URL('../', import.meta.resolve('doppler-gpu/compat')));
    for (const file of packageReceipt.files) {
      const installed = path.resolve(packageRoot, file.path);
      assert.ok(installed.startsWith(packageRoot), 'package inventory must remain within the installed root');
      assert.equal(hash(await fs.readFile(installed)), file.sha256, `installed package mismatch: ${file.path}`);
    }
  }
  assert.notEqual(Boolean(run.runtimeProfile), Boolean(run.runtimeConfigPath), 'Choose one explicit prepared configuration or named profile');
  const preparedBytes = run.runtimeConfigPath ? await fs.readFile(run.runtimeConfigPath) : null;
  if (preparedBytes) assert.equal(hash(preparedBytes), run.expectedRuntimeConfigSha256, 'Prepared configuration identity mismatch');
  const runtimeRoot = fileURLToPath(new URL('./config/runtime/', import.meta.resolve('doppler-gpu/compat')));
  const server = http.createServer(async (request, response) => {
    try {
      const name = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      if (name === '/prepared.json' && preparedBytes) {
        response.setHeader('Content-Type', 'application/json');
        response.end(preparedBytes);
        return;
      }
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
    installedFilesVerified: run.packageTarballPath ? packageReceipt.files.length : null,
    runtimeProfile: run.runtimeProfile ?? null,
    configurationSource: preparedBytes ? 'explicit-prepared' : 'named-profile',
    preparedConfigSha256: preparedBytes ? hash(preparedBytes) : null,
    rows: [], gpuErrors: [] };
  await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2), { flag: 'wx' });
  let model;
  try {
    const profileBaseUrl = `http://127.0.0.1:${server.address().port}`;
    const runtimeConfig = preparedBytes
      ? await applyRuntimeConfigFromUrl(`${profileBaseUrl}/prepared.json`)
      : await applyRuntimeProfile(run.runtimeProfile, { baseUrl: profileBaseUrl });
    model = await load({ url: pathToFileURL(run.modelDir).href }, { runtimeConfig, isolatedLoader: true });
    assert.equal(model.manifestHash, hash(modelBytes));
    assert.equal(model.resolvedArtifactVariantId, `sha256:${hash(modelBytes)}`);
    assert.equal(model.manifest.moeConfig.expertFormat, 'qwen3_5');
    assert.ok(getDevice(), 'public device export must identify the loaded host device');
    getDevice().addEventListener('uncapturederror', event => report.gpuErrors.push(event.error.message));
    report.loaded = true;
    report.artifactIdentity = model.manifest.artifactIdentity;
    report.resolvedRuntimeSession = model.advanced.getResolvedRuntimeSession();
    assert.equal(report.resolvedRuntimeSession.runtime.session.kvcache.kvDtype, expectedKvDtype);
    if (preparedBytes) {
      assert.deepEqual(report.resolvedRuntimeSession.kernelPath.definition,
        JSON.parse(preparedBytes).runtime.inference.kernelPath, 'Installed load must preserve the explicit prepared kernel path');
    }
    report.runtimeConfig = runtimeConfig;
    for (const id of run.sequences) {
      const input = inputs.sequences.find(row => row.id === id);
      model.resetGenerationState();
      const observed = [];
      const maxTokens = run.maxTokensBySequence?.[id] ?? 1;
      assert.ok(Number.isSafeInteger(maxTokens) && maxTokens > 0);
      const evidence = await model.generateWithEvidence(input.text, { inputIds: input.tokens,
        useChatTemplate: false, maxTokens, seed: 0, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1,
        onLogits(values, context) { assert(values.every(Number.isFinite)); observed.push({ values: Float32Array.from(values), ...context }); } });
      assert.equal(evidence.schema, 'doppler_generation_evidence/v1');
      assert.equal(evidence.tokenIds[0], reference.sequences[id].argmax[input.tokens.length - 1]);
      assert.equal(observed.length, evidence.tokenIds.length);
      assert.ok(evidence.resolution.resolvedExecutionId);
      assert.equal(evidence.resolution.resolvedArtifactVariantId, model.resolvedArtifactVariantId);
      const stepLogits = [];
      for (const [step, row] of observed.entries()) {
        assert.equal(row.tokenId, evidence.tokenIds[step]);
        assert.equal(row.inputTokenCount, input.tokens.length + step);
        const file = `${run.outputPath}.${id}.step${step}.f32`;
        const bytes = Buffer.from(row.values.buffer); await fs.writeFile(file, bytes);
        stepLogits.push({ step, inputTokenCount: row.inputTokenCount, tokenId: row.tokenId, file, sha256: hash(bytes) });
      }
      report.rows.push({ id, inputTokenCount: input.tokens.length, tokenId: evidence.tokenIds[0], tokenIds: evidence.tokenIds,
        kvStorageDtype: report.resolvedRuntimeSession.runtime.session.kvcache.kvDtype,
        prefillLogitsFile: stepLogits[0].file, prefillLogitsSha256: stepLogits[0].sha256, stepLogits, evidence });
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
