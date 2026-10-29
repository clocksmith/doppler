import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

const configPath = process.argv[2];
if (!configPath) throw new Error('Pass an explicit probe config JSON path');
const configBytes = await fs.readFile(configPath);
const config = JSON.parse(configBytes);
const here = path.dirname(fileURLToPath(import.meta.url));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
await fs.mkdir(config.outputDir, { recursive: false });
const sources = {};
for (const [name, file] of [
  ['runner', fileURLToPath(import.meta.url)], ['browser', path.join(here, 'layer_gpu_browser.js')],
  ['ortWebGpu', path.join(config.ortDist, 'ort.webgpu.min.mjs')],
  ['ortJsepWasm', path.join(config.ortDist, 'ort-wasm-simd-threaded.jsep.wasm')],
  ['prefill', path.join(config.modelDir, 'prefill.onnx')], ['decode', path.join(config.modelDir, 'decode.onnx')],
  ['fixtures', path.join(config.modelDir, 'fixtures.json')], ['cpuReceipt', path.join(config.modelDir, 'receipt.json')],
]) sources[name] = { path: file, sha256: sha(await fs.readFile(file)) };
await fs.writeFile(path.join(config.outputDir, 'identity.json'), JSON.stringify({ configSha256: sha(configBytes), sources }, null, 2));
const logs = [], errors = [];
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html' });
      response.end('<script type="module" src="/adapter/layer_gpu_browser.js"></script>');
      return;
    }
    if (pathname === '/config.json') {
      response.writeHead(200, { 'Content-Type': 'application/json' }); response.end(JSON.stringify(config)); return;
    }
    let file;
    for (const [prefix, root] of [['/model/', config.modelDir], ['/ort/', config.ortDist], ['/adapter/', here]]) {
      if (pathname.startsWith(prefix)) {
        file = path.resolve(root, decodeURIComponent(pathname.slice(prefix.length)));
        if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error('Path outside served root');
        break;
      }
    }
    if (!file) { response.writeHead(404); response.end(); return; }
    const stat = await fs.stat(file);
    const extension = path.extname(file);
    const type = ['.js', '.mjs'].includes(extension) ? 'text/javascript' :
      extension === '.json' ? 'application/json' : extension === '.wasm' ? 'application/wasm' : 'application/octet-stream';
    response.writeHead(200, { 'Content-Type': type, 'Content-Length': stat.size });
    createReadStream(file).on('error', error => response.destroy(error)).pipe(response);
  } catch (error) { response.writeHead(500); response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let context;
try {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, 'browser-profile'), {
    executablePath: config.browserExecutable, args: config.browserArgs, headless: true,
  });
  const page = await context.newPage();
  page.on('console', message => logs.push(`${message.type()}: ${message.text()}`));
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => globalThis.RESULT !== undefined, null, { timeout: config.timeoutMs });
  const result = await page.evaluate(() => globalThis.RESULT);
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify({ ...result, browserErrors: errors }, null, 2));
  if (!result.pass || errors.length) throw new Error(result.error || errors.join('\n'));
} finally {
  await context?.close();
  await new Promise(resolve => server.close(resolve));
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
