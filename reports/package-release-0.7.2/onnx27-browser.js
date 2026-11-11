// Runs the pinned upstream ONNX WebGPU worker without changing its kernels.
// Run from the pinned example checkout with its Playwright dependency installed.
import fs from 'node:fs/promises';
import { chromium } from 'playwright';

const origin = 'http://127.0.0.1:5174';
const report = { schema: 'doppler.onnx27-browser/v1', startedAt: new Date().toISOString(),
  upstreamCommit: 'aaa56b8c5accd10456819e18d3e27175afad69fc',
  modelRevision: '1030027de90ff1d54cefc66a23f253667723a938',
  userPrompt: 'Explain why the sky is blue and why sunsets look red. Use a clear paragraph with at least eighty words.',
  maxTokens: 64, warmupRuns: 1, timedRuns: 3, warnings: [], samples: [] };
const context = await chromium.launchPersistentContext('.e-chrome', {
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist'],
  ignoreDefaultArgs: ['--enable-unsafe-swiftshader'],
});
try {
  report.browser = context.browser().version();
  const page = context.pages()[0];
  page.on('console', message => {
    if (['warning', 'error'].includes(message.type())) report.warnings.push(message.text());
  });
  page.on('pageerror', error => report.warnings.push(String(error)));
  await page.route('**/*', route => new URL(route.request().url()).origin === origin
    ? route.continue() : route.abort());
  await page.goto(origin);
  await page.evaluate(() => {
    const worker = new Worker('/src/inference.worker.js', { type: 'module' });
    let requestId = 0;
    const pending = new Map();
    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') { console.log(data.message); return; }
      const task = pending.get(data.requestId);
      if (!task) return;
      clearTimeout(task.timer); pending.delete(data.requestId);
      data.type === 'error' ? task.reject(new Error(data.error)) : task.resolve(data.value);
    };
    worker.onerror = event => {
      for (const task of pending.values()) { clearTimeout(task.timer); task.reject(new Error(event.message)); }
      pending.clear();
    };
    window.eRequest = message => new Promise((resolve, reject) => {
      const id = ++requestId;
      const timer = setTimeout(() => { pending.delete(id); reject(new Error('Worker request exceeded 10 minutes')); }, 600000);
      pending.set(id, { resolve, reject, timer });
      worker.postMessage({ ...message, requestId: id });
    });
  });
  report.load = await page.evaluate(() => window.eRequest({ type: 'load', modelId: 'Qwen3.8-27B-GenAI-WebGPU-INT4' }));
  console.log(JSON.stringify({ loaded: true, load: report.load }));
  for (let i = 0; i < report.warmupRuns + report.timedRuns; i++) {
    const result = await page.evaluate(({ prompt, maxTokens }) => window.eRequest({
      type: 'generate', messages: [{ role: 'user', content: prompt }],
      maxTokens, temperature: 0, enableThinking: false,
    }), { prompt: report.userPrompt, maxTokens: report.maxTokens });
    report.samples.push({ warmup: i < report.warmupRuns, ...result });
    await fs.writeFile('e-onnx27-result.json', JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ sample: i, tokens: result.completionTokens, elapsedMs: result.elapsedMs, text: result.text }));
  }
  report.passed = report.load.cpuFallbackDisabled && !report.load.gpu.isFallbackAdapter
    && report.samples.every(sample => sample.completionTokens === report.maxTokens
      && sample.steps.every(step => step.allStateOnGPU && step.stateTensorsRetained === 128));
} catch (error) {
  report.error = String(error.stack ?? error); process.exitCode = 1;
} finally {
  await context.close();
  await fs.writeFile('e-onnx27-result.json', `${JSON.stringify(report, null, 2)}\n`);
}
