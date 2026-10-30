import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const config = JSON.parse(await fs.readFile(process.argv[2]));
await fs.mkdir(config.outputDir, { recursive: false });
const allCases = JSON.parse(await fs.readFile(path.join(config.probeDir, 'probes.json')));
const cases = config.probeNames ? allCases.filter(test => config.probeNames.includes(test.name)) : allCases;
if (config.probeNames && cases.length !== config.probeNames.length) throw new Error('Requested probe names differ from generated graph inventory');
const sha = data => createHash('sha256').update(data).digest('hex');
const identity = { script: sha(await fs.readFile(fileURLToPath(import.meta.url))),
  ortWebGpu: sha(await fs.readFile(path.join(config.ortDist, 'ort.webgpu.min.mjs'))),
  ortWasm: sha(await fs.readFile(path.join(config.ortDist, 'ort-wasm-simd-threaded.jsep.wasm'))),
  graphs: await Promise.all(cases.map(async test => ({ name: test.name, file: test.file,
    sha256: sha(await fs.readFile(path.join(config.probeDir, test.file))), inputs: test.inputs }))) };
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') { response.writeHead(200, { 'Content-Type': 'text/html' }); response.end('op admission'); return; }
    const prefix = pathname.startsWith('/ort/') ? '/ort/' : '/probes/';
    const root = prefix === '/ort/' ? config.ortDist : config.probeDir;
    const file = path.resolve(root, decodeURIComponent(pathname.slice(prefix.length)));
    if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error('outside served root');
    const stat = await fs.stat(file), extension = path.extname(file);
    response.writeHead(200, { 'Content-Length': stat.size,
      'Content-Type': extension === '.mjs' ? 'text/javascript' : extension === '.wasm' ? 'application/wasm' : 'application/octet-stream' });
    createReadStream(file).on('error', error => response.destroy(error)).pipe(response);
  } catch (error) { response.writeHead(500); response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const logs = [];
let context;
try {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, 'browser-profile'), {
    executablePath: config.browserExecutable, args: config.browserArgs, headless: true });
  const page = await context.newPage();
  page.on('console', message => logs.push(message.text()));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async ({ config, cases }) => {
    const ort = await import('/ort/ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = '/ort/'; ort.env.wasm.numThreads = 1;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter?.info.vendor.toLowerCase().includes(config.requiredVendor)) throw new Error('wrong adapter');
    const device = await adapter.requestDevice();
    ort.env.webgpu.device = device;
    const actualDevices = new Set();
    const bounded = async (operation, phase) => {
      let timer;
      try { return await Promise.race([operation, new Promise((resolve, reject) => {
        timer = setTimeout(() => reject(new Error('micro observation timeout: ' + phase)), config.gpuInspectionTimeoutMs);
      })]); } finally { clearTimeout(timer); }
    };
    const attempts = [];
    try {
      for (const test of cases) {
        let session, output, inspection;
        const feeds = {};
        try {
          session = await ort.InferenceSession.create(`/probes/${test.file}`, { executionProviders: ['webgpu'],
            graphOptimizationLevel: 'disabled', logSeverityLevel: 0, logVerbosityLevel: 1,
            ...(config.inspectGpuTensor === true ? { preferredOutputLocation: { output: 'gpu-buffer' } } : {}),
            extra: { session: { disable_cpu_ep_fallback: '1' } } });
          if (config.admissionOnly === true) {
            attempts.push({ name: test.name, admitted: true, executed: false, sourceNodes: test.sourceNodes });
            continue;
          }
          for (const input of test.inputs) {
            const size = input.shape.reduce((a, b) => a * b, 1);
            const values = input.values ?? new Array(size).fill(0);
            const data = input.type === 'int64' ? BigInt64Array.from(values, BigInt) :
              input.type === 'int32' ? Int32Array.from(values) : Float32Array.from(values);
            feeds[input.name] = new ort.Tensor(input.type, data, input.shape);
          }
          output = await session.run(feeds);
          if (config.inspectGpuTensor === true) {
            if (!Number.isSafeInteger(config.gpuInspectionTimeoutMs) || config.gpuInspectionTimeoutMs <= 0) throw new Error('Declare positive gpuInspectionTimeoutMs');
            const actual = await ort.env.webgpu.device;
            actualDevices.add(actual);
            const tensor = output.output;
            if (tensor.location !== 'gpu-buffer' || tensor.type !== 'float32') throw new Error('Micro observer requires actual F32 GPU output');
            inspection = { sourceLocation: tensor.location, requestedDeviceHonored: actual === device,
              actualAdapter: { vendor: actual.adapterInfo.vendor, architecture: actual.adapterInfo.architecture },
              gpuErrors: [], phases: [] };
            actual.addEventListener('uncapturederror', event => inspection.gpuErrors.push(event.error.message));
            const phase = name => { inspection.phases.push(name); console.log('observation-micro ' + name); };
            const bytes = 4 * test.expected.length;
            const target = actual.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            try {
              phase('direct-copy-submit');
              const commands = actual.createCommandEncoder();
              commands.copyBufferToBuffer(tensor.gpuBuffer, 0, target, 0, bytes);
              actual.queue.submit([commands.finish()]);
              phase('direct-copy-map');
              await bounded(target.mapAsync(GPUMapMode.READ), 'direct-copy-map');
              inspection.directCopy = Array.from(new Float32Array(target.getMappedRange()));
              target.unmap(); phase('direct-copy-complete');
            } catch (error) { inspection.directCopyError = String(error); }
            finally { target.destroy(); }
            try {
              phase('empty-error-scope');
              actual.pushErrorScope('validation');
              const error = await bounded(actual.popErrorScope(), 'empty-error-scope');
              inspection.emptyErrorScope = error?.message ?? null;
              phase('empty-error-scope-complete');
            } catch (error) { inspection.emptyErrorScopeError = String(error); }
            phase('public-tensor-getData');
            const downloaded = await bounded(tensor.getData(), 'public-tensor-getData');
            inspection.publicDownload = Array.from(downloaded);
            inspection.locationAfterPublicDownload = tensor.location;
            phase('public-tensor-getData-complete');
          }
          const data = Array.from(await output.output.getData(), Number);
          attempts.push({ name: test.name, admitted: true, output: data,
            expected: test.expected, inspection, exact: data.length === test.expected.length && data.every((value, index) => value === test.expected[index]) });
        } catch (error) { attempts.push({ name: test.name, admitted: false, inspection, sourceNodes: test.sourceNodes, error: String(error) }); }
        finally {
          for (const tensor of Object.values(output ?? {})) tensor.dispose();
          for (const tensor of Object.values(feeds)) tensor.dispose();
          await session?.release();
        }
      }
    } finally { for (const actual of actualDevices) if (actual !== device) actual.destroy(); device.destroy(); }
    return { schema: 'doppler.pinned-ort-gpu-op-admission.v1', scope: 'Exact-dtype micrographs, no CPU fallback, no model qualification',
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture }, attempts };
  }, { config, cases });
  result.identity = identity;
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await context?.close(); await new Promise(resolve => server.close(resolve));
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
