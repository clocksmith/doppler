// Run from the repository root: node artifacts/.../doppler.js MODEL_DIR INPUT_JSON OUTPUT_JSON
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { doppler } from '../../src/index.js';
import { getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
const [modelDir, inputPath, outputPath, policyPath] = process.argv.slice(2);
if (!outputPath) throw new Error('Expected MODEL_DIR INPUT_JSON OUTPUT_JSON');
const policy = JSON.parse(await fs.readFile(policyPath ?? new URL('./policy.json', import.meta.url)));
const input = JSON.parse(await fs.readFile(inputPath));
if (JSON.stringify(input.source) !== JSON.stringify(policy.source)) throw new Error('Prompt source mismatch');
if (JSON.stringify(input.prompts.map(prompt => prompt.tokenIds.length)) !== JSON.stringify(policy.shared.promptTokenCounts)) {
  throw new Error('Prompt lengths mismatch');
}
const manifestBytes = await fs.readFile(path.join(modelDir, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
if (manifest.artifactIdentity.sourceRevision !== policy.source.revision) throw new Error('Source revision mismatch');
if (JSON.stringify(manifest.eos_token_id) !== JSON.stringify(policy.shared.eosTokenIds)) throw new Error('EOS IDs differ');
const result = { engine: 'doppler', surface: 'node-webgpu', nodeVersion: process.version,
  manifestHash: createHash('sha256').update(manifestBytes).digest('hex'), policy, runs: [], complete: false };
let model;
try {
  const start = performance.now();
  model = await doppler.load({url: pathToFileURL(path.resolve(modelDir) + path.sep).href}, {runtimeConfig: policy.engines.doppler.runtimeConfig});
  result.modelLoadMs = performance.now() - start;
  result.device = getKernelCapabilities();
  result.loadedMemory = {process: process.memoryUsage(), pool: getBufferPool().getStats(), labels: getBufferPool().getLabelStats()};
  for (const prompt of input.prompts) {
    for (let i = 0; i < policy.shared.warmupRuns + policy.shared.timedRuns; i++) {
      model.resetGenerationState();
      const began = performance.now(), tokenTimes = [];
      const evidence = await model.generateWithEvidence('', {
        inputIds: prompt.tokenIds, maxTokens: policy.shared.maxTokens,
        temperature: policy.shared.temperature, topK: policy.shared.topK, topP: policy.shared.topP,
        repetitionPenalty: policy.shared.repetitionPenalty, useChatTemplate: false,
        suppressTokenIds: policy.shared.suppressTokenIds,
        onToken() {tokenTimes.push(performance.now() - began);}
      });
      const totalRunMs = performance.now() - began;
      result.runs.push({promptTokenIds: prompt.tokenIds, warmup: i < policy.shared.warmupRuns, totalRunMs,
        firstTokenMs: tokenTimes[0] ?? null, tokenTimes, evidence,
        memory: {process: process.memoryUsage(), pool: getBufferPool().getStats()}});
      await fs.writeFile(outputPath, JSON.stringify(result, null, 2));
      console.log(JSON.stringify({promptTokens: prompt.tokenIds.length, run: i, tokens: evidence.tokenIds.length, totalRunMs}));
    }
  }
  result.complete = true;
} catch (error) { result.error = {message: error.message, stack: error.stack}; process.exitCode = 1; }
finally {
  try {await model?.unload();} catch (error) {result.cleanupError = error.message; result.complete = false; process.exitCode = 1;}
  if (getDevice()) { result.unloadedPool = getBufferPool().getStats(); getDevice().destroy(); }
  await fs.writeFile(outputPath, JSON.stringify(result, null, 2));
}
