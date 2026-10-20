import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { resetLinearAttentionRuntime } from '../../src/inference/pipelines/text/linear-attention.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const modelDirectory = process.env.DOPPLER_PREPARED_MODEL_DIR;
if (!modelDirectory) {
  assert(!process.argv.includes('--require-gpu'), 'Set DOPPLER_PREPARED_MODEL_DIR');
  console.log('linear-prepared-sequence-physical: skipped (no provisioned model)');
} else {
  const bytes = await fs.readFile(path.join(modelDirectory, 'manifest.json'));
  const manifest = JSON.parse(bytes);
  const runtimeConfig = process.env.DOPPLER_PREPARED_CONFIG
    ? JSON.parse(await fs.readFile(process.env.DOPPLER_PREPARED_CONFIG, 'utf8')).request.runtimeConfig
    : {};
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason);
  const errors = [];
  getDevice().addEventListener('uncapturederror', event => errors.push(event.error.message));
  let pipeline;
  const report = { modelId: manifest.modelId,
    manifestHash: createHash('sha256').update(bytes).digest('hex'), provider: probe.providerReceipt,
    scope: 'Node physical fresh-conversation reset and cancellation parity; not performance or general quality', rows: [] };
  try {
    const storage = createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(modelDirectory)).href, manifest);
    pipeline = await createPipeline(manifest, { runtimeConfig, storage });
    const options = { useChatTemplate: false, maxTokens: 32, temperature: 0, topK: 1,
      topP: 1, repetitionPenalty: 1, seed: 0 };
    const generate = async (prompt, fullReset) => {
      pipeline.reset();
      if (fullReset) pipeline.linearAttentionRuntime = resetLinearAttentionRuntime(pipeline.linearAttentionRuntime);
      const result = await pipeline.generateTokenIds(prompt, options);
      return Array.from(result.tokenIds);
    };
    const prompts = ['The capital of France is', 'Write a JavaScript function that adds two numbers.',
      'Three movie genres are', 'The capital of France is'];
    for (const prompt of prompts) {
      const expected = await generate(prompt, true);
      const prepared = new Map([...pipeline.linearAttentionRuntime.layers].map(([index, layer]) => [index, layer.convWeightGPU]));
      assert(prepared.size > 0);
      const actual = await generate(prompt, false);
      assert.deepEqual(actual, expected, 'retained preparation matches full state reconstruction');
      for (const [index, buffer] of prepared) {
        assert.equal(pipeline.linearAttentionRuntime.layers.get(index).convWeightGPU, buffer,
          'prepared buffer reused across conversation reset');
      }
      report.rows.push({ prompt, tokenIds: actual, exactFullResetParity: true, reusedLayers: prepared.size });
    }
    const abort = new AbortController();
    let chunks = 0;
    try {
      for await (const _chunk of pipeline.generate(prompts[1], { ...options, signal: abort.signal })) {
        chunks++;
        abort.abort();
      }
    } catch (error) {
      assert(abort.signal.aborted && error.name === 'AbortError', String(error));
    }
    assert(chunks > 0);
    assert.equal(pipeline.isGenerating, false);
    const afterCancel = await generate(prompts[0], false);
    assert.deepEqual(afterCancel, report.rows[0].tokenIds);
    report.cancellation = { chunks, nextRequestMatchesFresh: true };
    await getDevice().queue.onSubmittedWorkDone();
    assert.deepEqual(errors, []);
    report.passed = true;
  } finally {
    await pipeline?.unload();
    destroyDevice();
    releaseNodeWebGPU();
  }
  console.log(JSON.stringify(report, null, 2));
}
