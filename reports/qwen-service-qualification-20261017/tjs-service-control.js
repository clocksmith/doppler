import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';
import { formatChatMessages } from '../../src/inference/pipelines/text/chat-format.js';

// Run from the Doppler root. This is a bounded quality control, not a speed claim.
const root = process.cwd();
const suiteFile = process.argv[2];
const output = process.argv[3];
const packageMetadata = JSON.parse(await fs.readFile(path.join(root, 'node_modules/@huggingface/transformers/package.json'), 'utf8'));
if (packageMetadata.version !== '4.2.0') throw new Error('This control requires pinned Transformers.js 4.2.0');
const suiteBytes = await fs.readFile(suiteFile);
const suite = JSON.parse(suiteBytes);
const system = "You're Reploid. Answer briefly and directly; expand when asked. Model: Qwen 3.5 2B.";
const tasks = suite.tasks.map(task => ({ ...task, promptText: formatChatMessages([
  { role: 'system', content: system }, { role: 'user', content: task.prompt },
], 'qwen', { thinking: false }) }));
const report = { schema: 'doppler.tjs-service-control/v1', startedAt: new Date().toISOString(),
  model: 'onnx-community/Qwen3.5-2B-ONNX-OPT', dtype: 'q4f16', system, packageVersion: packageMetadata.version,
  suiteDigest: createHash('sha256').update(suiteBytes).digest('hex'),
  generation: { do_sample: false, max_new_tokens: 64, repetition_penalty: 1 },
  scope: 'Same rendered Reploid single-turn prompts; different quantization formats; quality control, not independent application retention.',
  results: [], logs: [], passed: false };
let server, browser, timer;
try {
  server = await createStaticFileServer({ rootDir: root, host: '127.0.0.1', port: 0,
    staticMounts: [{ urlPrefix: '/models', rootDir: path.join(root, '.cache/qwen-comparator-relay') }] });
  browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
  report.browser = browser.version();
  const page = await browser.newPage();
  page.on('console', message => report.logs.push({ type: message.type(), text: message.text() }));
  page.on('pageerror', error => report.logs.push({ type: 'pageerror', text: error.message }));
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.origin !== server.baseUrl) return route.abort();
    if (url.pathname === '/service-control') return route.fulfill({ contentType: 'text/html', body: `<!doctype html><script type="importmap">{"imports":{"onnxruntime-common":"/node_modules/@huggingface/transformers/node_modules/onnxruntime-common/dist/esm/index.js","onnxruntime-web":"/node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs","onnxruntime-web/webgpu":"/node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs"}}</script>` });
    return route.continue();
  });
  await page.goto(server.baseUrl + '/service-control');
  await page.exposeFunction('retainControlRow', async row => {
    report.results.push(row);
    await fs.writeFile(output, JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ id: row.id, answer: row.answer, passed: row.passed }));
  });
  const execution = page.evaluate(async ({ tasks, report }) => {
    const tjs = await import('/node_modules/@huggingface/transformers/dist/transformers.web.min.js');
    const ort = await import('onnxruntime-web/webgpu');
    ort.env.wasm.proxy = false; ort.env.wasm.numThreads = 1;
    const wasmPaths = '/node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/';
    ort.env.wasm.wasmPaths = wasmPaths;
    Object.assign(tjs.env.backends.onnx.wasm, { proxy: false, numThreads: 1, wasmPaths });
    Object.assign(tjs.env, { allowLocalModels: true, allowRemoteModels: false, localModelPath: '/models/' });
    const adapter = await navigator.gpu.requestAdapter();
    const hardware = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, adapter.info[key]]));
    const start = performance.now();
    const tokenizer = await tjs.AutoTokenizer.from_pretrained(report.model);
    const model = await tjs.Qwen3_5ForConditionalGeneration.from_pretrained(report.model,
      { device: 'webgpu', dtype: report.dtype, session_options: { executionProviders: ['webgpu'], enableMemPattern: false } });
    const loadMs = performance.now() - start;
    for (const task of tasks) {
      const inputs = tokenizer([task.promptText], { add_special_tokens: false, padding: true, truncation: true });
      const promptIds = Array.from(inputs.input_ids.data, Number);
      let firstTokenMs = null, first = true;
      const began = performance.now();
      const streamer = { put() { if (first) { first = false; return; } if (firstTokenMs === null) firstTokenMs = performance.now() - began; }, end() {} };
      const output = await model.generate({ ...inputs, ...report.generation, streamer });
      const elapsedMs = performance.now() - began;
      const ids = Array.from(output.data, Number).slice(promptIds.length);
      const answer = tokenizer.decode(ids, { skip_special_tokens: true }).trim();
      let actual = answer;
      if ('expectedJson' in task) { try { actual = JSON.parse(answer); } catch {} }
      const expected = task.expectedJson ?? task.expected;
      const passed = JSON.stringify(actual) === JSON.stringify(expected);
      await globalThis.retainControlRow({ id: task.id, promptText: task.promptText, promptIds, ids, answer, expected, passed, firstTokenMs, elapsedMs });
    }
    await model.dispose();
    return { hardware, loadMs, version: tjs.VERSION };
  }, { tasks, report });
  Object.assign(report, await Promise.race([execution, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('600-second control budget exceeded')), 600000);
  })]));
  report.passed = report.results.length === tasks.length && report.results.every(row => row.passed);
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  clearTimeout(timer);
  await browser?.close();
  await server?.close();
  report.finishedAt = new Date().toISOString();
  await fs.writeFile(output, JSON.stringify(report, null, 2));
}
process.exitCode = report.error ? 1 : 0;
