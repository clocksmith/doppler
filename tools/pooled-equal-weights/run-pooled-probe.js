import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';

const configBytes = await fs.readFile(process.argv[2], 'utf8');
const config = JSON.parse(configBytes);
const browserEntry = config.browserEntry ?? 'probe-browser.js';
if (!/^[a-z0-9-]+\.js$/.test(browserEntry)) throw new Error('Invalid browser observation entrypoint');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const here = path.dirname(fileURLToPath(import.meta.url));
await fs.mkdir(config.outputDir, { recursive: false });
await fs.cp(path.join(config.pooledRoot, 'engine'), path.join(config.outputDir, 'vendor', 'engine'), { recursive: true, dereference: true });
const identity = [];
async function inventory(dir) {
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) await inventory(file);
    else identity.push({ path: path.relative(config.outputDir, file), sha256: sha(await fs.readFile(file)) });
  }
}
await inventory(path.join(config.outputDir, 'vendor'));
const contractBytes = await fs.readFile(config.contractPath, 'utf8'), contract = JSON.parse(contractBytes);
await fs.writeFile(path.join(config.outputDir, 'source-identity.json'), JSON.stringify({ configSha256: sha(configBytes), contractSha256: sha(contractBytes), engineFiles: identity,
  nativeKvShaderSha256: config.nativeKvShader ? sha(await fs.readFile(config.nativeKvShader)) : null,
  adapterFiles: await Promise.all(
  ['run-pooled-probe.js', 'probe-browser.js', 'pooled-canonical.js', 'verify-pooled-gpu.js', 'shared-input.js', 'capture-pooled-boundaries.js',
    'rne-probe-browser.js', 'kv-cache-rne-prototype.wgsl'].map(async name => ({ name, sha256: sha(await fs.readFile(path.join(here, name))) }))) }, null, 2));
const logs = [], errors = [];
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/') { response.setHeader('Content-Type', 'text/html'); response.end(`<script type="module" src="/adapter/${browserEntry}"></script>`); return; }
    if (url.pathname === '/probe-config.json') { response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify({ ...config, contract })); return; }
    const routes = [['/vendor/', path.join(config.outputDir, 'vendor')], ['/canonical/', config.canonicalDir], ['/adapter/', here]];
    let file;
    if (url.pathname === '/model.gguf') file = config.ggufPath;
    else if (url.pathname === '/native-kv.wgsl' && config.nativeKvShader) file = config.nativeKvShader;
    else for (const [prefix, root] of routes) if (url.pathname.startsWith(prefix)) {
      const relative = decodeURIComponent(url.pathname.slice(prefix.length));
      file = path.resolve(root, relative);
      if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error('Path outside served root');
      break;
    }
    if (!file) { response.writeHead(404); response.end(); return; }
    const stat = await fs.stat(file);
    const range = request.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    if (request.headers.range && !range) throw new Error('Unsupported byte range');
    const start = range ? Number(range[1]) : 0, end = range ? Math.min(Number(range[2]), stat.size - 1) : stat.size - 1;
    if (start > end || end >= stat.size) throw new Error('Invalid byte extent');
    const mime = path.extname(file) === '.js' ? 'text/javascript' : path.extname(file) === '.json' ? 'application/json' : 'application/octet-stream';
    response.writeHead(range ? 206 : 200, { 'Content-Type': mime, 'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes',
      ...(range ? { 'Content-Range': `bytes ${start}-${end}/${stat.size}` } : {}) });
    createReadStream(file, { start, end }).on('error', error => response.destroy(error)).pipe(response);
  } catch (error) { if (!response.headersSent) response.writeHead(500); response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let context;
try {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, 'browser-profile'), {
    executablePath: config.browserExecutable, args: config.browserArgs, headless: true });
  const page = await context.newPage();
  page.on('console', message => { logs.push(`${message.type()}: ${message.text()}`); console.log(message.text()); });
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => globalThis.RESULT !== undefined, null, { timeout: config.timeoutMs });
  const result = await page.evaluate(() => globalThis.RESULT);
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify({ ...result, browserErrors: errors }, null, 2));
  if (result.error || errors.length || result.pass === false) throw new Error(result.error || errors.join('\n') || 'Observation acceptance failed');
} finally {
  await context?.close(); await new Promise(resolve => server.close(resolve));
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
