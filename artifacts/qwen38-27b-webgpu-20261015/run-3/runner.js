import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { doppler } from '../../src/index.js';

const [modelDir, outputDir] = process.argv.slice(2);
if (!modelDir || !outputDir) throw new Error('Usage: node run.js MODEL_DIR OUTPUT_DIR');
await fs.mkdir(outputDir, { recursive: true });
const manifest = JSON.parse(await fs.readFile(path.join(modelDir, 'manifest.json'), 'utf8'));
const { runtimeConfig, options } = JSON.parse(await fs.readFile(new URL('./policy.json', import.meta.url), 'utf8'));
const record = { source: manifest.artifactIdentity, modelId: manifest.modelId, runtimeConfig, options, runs: [] };
const save = () => fs.writeFile(path.join(outputDir, 'result.json'), JSON.stringify(record, null, 2));
let model;
try {
  const start = performance.now();
  model = await doppler.load({ url: pathToFileURL(path.resolve(modelDir) + path.sep).href }, { runtimeConfig,
    onProgress(progress) { console.log(JSON.stringify({ phase: 'load', progress })); } });
  record.loadMs = performance.now() - start;
  console.log(JSON.stringify({ phase: 'loaded', loadMs: record.loadMs }));
  const messages = [{ role: 'user', content: 'What is 17 + 25? Answer with the number only.' }];
  for (const content of [null, 'What number did you just give me? Answer with the number only.']) {
    if (content) messages.push({ role: 'user', content });
    const began = performance.now();
    const result = await model.chatText(messages, options);
    record.runs.push({ messages: structuredClone(messages), elapsedMs: performance.now() - began, result });
    await save();
    console.log(JSON.stringify({ phase: 'answer', content: result.content, usage: result.usage }));
    messages.push({ role: 'assistant', content: result.content });
  }
} catch (error) {
  record.error = { message: error.message, stack: error.stack };
  await save();
  throw error;
} finally {
  await model?.unload();
}
