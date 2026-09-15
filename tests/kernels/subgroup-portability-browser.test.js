import path from 'node:path';
import { chromium } from 'playwright';
import { createStaticFileServer } from '../../src/tooling/node-browser-command-runner.js';

const server = await createStaticFileServer({ rootDir: path.resolve(import.meta.dirname, '../..'), port: 0 });
let browser;
const args = ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
  '--disable-dawn-features=disallow_unsafe_apis', '--ignore-gpu-blocklist'];
if (process.platform === 'linux') args.push('--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface');
if (process.platform === 'darwin') args.push('--use-angle=metal');
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true, args, timeout: 60000 });
  const page = await browser.newPage();
  await page.goto(`${server.baseUrl}/tests/capsule/browser-capsule-qualification.html`);
  const result = await page.evaluate(async () => {
    const { runSubgroupPortabilityCases } = await import('/tests/helpers/subgroup-portability.js');
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('Physical WebGPU adapter required.');
    const info = adapter.info;
    const identity = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
    if (info.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(JSON.stringify(identity))) {
      throw new Error(`Physical adapter required: ${JSON.stringify(identity)}`);
    }
    if (!navigator.gpu.wgslLanguageFeatures.has('subgroup_id')) throw new Error('subgroup_id is unavailable.');
    const device = await adapter.requestDevice({ requiredFeatures: ['subgroups'] });
    const errors = [];
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    try {
      const sources = {};
      for (const [key, file] of Object.entries({ stats: 'rmsnorm_stats_subgroups.wgsl',
        portableStats: 'rmsnorm_stats.wgsl', attention: 'attention_decode_subgroup.wgsl' })) {
        const response = await fetch(`/src/gpu/kernels/${file}`);
        if (!response.ok) throw new Error(`Cannot fetch ${file}: ${response.status}`);
        sources[key] = await response.text();
      }
      const counts = await runSubgroupPortabilityCases(device, sources);
      if (errors.length) throw new Error(errors.join('\n'));
      return { passed: true, ...counts, adapter: identity, languageFeatures: [...navigator.gpu.wgslLanguageFeatures] };
    } finally { device.destroy(); }
  });
  console.log(JSON.stringify({ test: 'subgroup-portability-browser', browserVersion: browser.version(), ...result,
    evidence: 'Physical browser operator parity, original and instrumented data-index permutations; no model qualification.' }));
} finally {
  await browser?.close();
  await server.close();
}
