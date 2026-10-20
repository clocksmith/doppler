import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { loadLoRAAdapterForPipeline } from '../../src/client/runtime/lora.js';
import { probeNodeGPU } from '../../tests/helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
if (!process.env.MODEL_DIR) {
  console.log('speculative-generation-physical: skipped; set MODEL_DIR to a local verified model');
} else {
const directory = process.env.MODEL_DIR, manifest = JSON.parse(await fs.readFile(path.join(directory, 'manifest.json')));
assert.equal((await probeNodeGPU({ installFileFetchShim: true })).ready,true);
const pipeline = await createPipeline(manifest, { storage: createNodeFileArtifactStorageContext(pathToFileURL(directory).href, manifest), runtimeConfig: { inference: { session: { kvcache: { maxSeqLen: 512 } } } } });
const adapter = process.env.LORA_CONFIG ? JSON.parse(await fs.readFile(process.env.LORA_CONFIG)) : null;
const sourceFiles = {};
for (const name of ['src/inference/pipelines/text/generator/speculative-runtime.js', 'src/inference/pipelines/text/linear-attention/state-checkpoint.js', 'src/inference/pipelines/text/generator/decode-runtime.js', 'src/inference/pipelines/text/generator/prefill-runtime.js', 'src/inference/pipelines/text/generation-request.js', 'src/inference/pipelines/text/generator-runtime.js', 'src/config/schema/speculation-self.schema.js']) {
 sourceFiles[name] = createHash('sha256').update(await fs.readFile(new URL('../../' + name, import.meta.url))).digest('hex');
}
try {
if (adapter) {
 await loadLoRAAdapterForPipeline(pipeline, adapter.manifest, { weightsLayout: adapter.weightsLayout,
  fetchUrl: async () => {
   const bytes = await fs.readFile(process.env.LORA_WEIGHTS);
   assert.equal('sha256:' + createHash('sha256').update(bytes).digest('hex'), adapter.artifact.hash);
   return bytes;
  } });
 assert.equal(pipeline.getActiveLoRA().identity.digest, adapter.identity);
}
console.log('SPECULATION_IDENTITY=' + JSON.stringify({ modelId: manifest.modelId,
 manifestHash: 'sha256:' + createHash('sha256').update(await fs.readFile(path.join(directory,'manifest.json'))).digest('hex'),
 sourceFiles, adapter: adapter?.identity ?? null, maxSeqLen: 512, tokens: 3, minMatchTokens: 3, maxMatchTokens: 4, maxRejectedBlocks: 1,
 temperature: 0, topK: 1, topP: 1, maxTokens: 40, repetitionPenalty: 1, repetitionPenaltyWindow: 1, useChatTemplate: true }));
const options = { maxTokens: 40, maxSeqLen: 512, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, repetitionPenaltyWindow: 1, useChatTemplate: true };
const prompts = ['Repeat exactly: The red fox jumps over the blue box. The red fox jumps over the blue box. The red fox jumps over the blue box.', 'What is two plus three?', 'Write a JSON array of four objects with keys name and city.'];
const report = [];
 for (const prompt of prompts) {
  const results = [];
  for (const mode of ['none','lookup','none']) {
   await pipeline.reset();
   const tokens = [], start = performance.now();
   for await (const token of pipeline.generateTokens([{role:'user',content:prompt}], { ...options, speculation: { mode, tokens: 3, maxMatchTokens: 4, minMatchTokens: 3, maxRejectedBlocks: 1 } })) tokens.push(token);
   results.push({ mode, tokens, ms:performance.now()-start, stats: { decodeMode: pipeline.stats.decodeMode, proposed: pipeline.stats.speculationAttempts, accepted: pipeline.stats.speculationAccepted, stopReason: pipeline.stats.stopReason } });
  }
  report.push({prompt,results});
  console.log('SPECULATION_RESULT='+JSON.stringify(report.at(-1)));
  assert.deepEqual(results[1].tokens,results[0].tokens);assert.deepEqual(results[2].tokens,results[0].tokens);
 }
 const expected = report[0].results[0].tokens;
 const abort = new AbortController(), partial = [];
 await pipeline.reset();
 for await (const token of pipeline.generateTokens([{role:'user',content:prompts[0]}], { ...options, speculation: { mode: 'lookup', tokens: 3, maxMatchTokens: 4, minMatchTokens: 3, maxRejectedBlocks: 1 }, signal: abort.signal })) {
   partial.push(token); if (partial.length === 4) abort.abort();
 }
 assert.deepEqual(partial, expected.slice(0, 4));
 await pipeline.reset();
 const reused = [];
 for await (const token of pipeline.generateTokens([{role:'user',content:prompts[0]}], options)) reused.push(token);
 assert.deepEqual(reused, expected);
 console.log('SPECULATION_LIFECYCLE=cancelled-and-reused');
} finally { await pipeline.unload(); destroyDevice(); releaseNodeWebGPU(); }

}
