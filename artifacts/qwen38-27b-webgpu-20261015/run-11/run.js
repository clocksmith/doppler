import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import { doppler } from '../../src/index.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
import { getDevice } from '../../src/gpu/device.js';

const [modelDir, outputDir] = process.argv.slice(2);
if (!modelDir || !outputDir) throw new Error('Usage: node run.js MODEL_DIR OUTPUT_DIR');
await fs.mkdir(outputDir, { recursive: true });
const manifest = JSON.parse(await fs.readFile(path.join(modelDir, 'manifest.json'), 'utf8'));
const { runtimeConfig, options } = JSON.parse(await fs.readFile(new URL('./policy.json', import.meta.url), 'utf8'));
const record = { source: manifest.artifactIdentity, modelId: manifest.modelId, runtimeConfig, options, runs: [], complete: false };
record.memory = [];
const observeMemory = (phase) => {
  const pool = getBufferPool(device);
  record.memory.push({ phase, ...process.memoryUsage(), pool: pool.getStats(),
    labels: pool.getLabelStats().sort((a, b) => b.bytes - a.bytes).slice(0, 40) });
};
const save = () => fs.writeFile(path.join(outputDir, 'result.json'), JSON.stringify(record, null, 2));
let model;
let device;
try {
  const start = performance.now();
  model = await doppler.load({ url: pathToFileURL(path.resolve(modelDir) + path.sep).href }, { runtimeConfig,
    onProgress(progress) { console.log(JSON.stringify({ phase: 'load', progress })); } });
  record.loadMs = performance.now() - start;
  device = getDevice();
  observeMemory('loaded');
  console.log(JSON.stringify({ phase: 'loaded', loadMs: record.loadMs }));
  async function answer(messages) {
    const began = performance.now();
    const result = await model.chatText(messages, options);
    observeMemory(`answer-${record.runs.length + 1}`);
    record.runs.push({ messages: structuredClone(messages), elapsedMs: performance.now() - began, result });
    await save();
    console.log(JSON.stringify({ phase: 'answer', content: result.content, usage: result.usage }));
    assert.equal(result.evidence.stats.stopReason, 'stop-token');
    assert(!result.content.includes('\uFFFD'));
    return result.content;
  }
  const messages = [{ role: 'user', content: 'Use only this fictional catalog. Cedar: English, 100 minutes, gentle comedy, no violence. Storm: English, 130 minutes, violent thriller. Lumiere: French, 95 minutes, gentle comedy. We require English, no violence, and at most 110 minutes. Pick exactly one title and briefly explain which requirements it meets.' }];
  const first = await answer(messages);
  assert(first.includes('Cedar'));
  messages.push({ role: 'assistant', content: first }, { role: 'user', content: 'What is the runtime of the movie you picked? Answer in one sentence.' });
  assert(/100/.test(await answer(messages)));
  assert.equal((await answer([{ role: 'user', content: 'What is 17 + 25? Answer with the number only.' }])).trim(), '42');
  record.answersPassed = true;
} catch (error) {
  record.error = { message: error.message, stack: error.stack };
  await save();
} finally {
  console.log(JSON.stringify({ phase: 'unloading' }));
  try {
    await model?.unload();
    if (device) {
      observeMemory('unloaded');
      assert.equal(getBufferPool(device).getStats().activeBuffers, 0);
    }
    record.unloaded = true;
  } catch (error) {
    record.cleanupError = { message: error.message, stack: error.stack };
  } finally {
    record.complete = record.answersPassed === true && record.error == null && record.cleanupError == null;
    await save();
    console.log(JSON.stringify({ phase: 'unloaded', complete: record.complete }));
    device?.destroy();
  }
}
process.exit(record.complete ? 0 : 1);
