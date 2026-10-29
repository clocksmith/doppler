import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const configPath = process.argv[2];
if (!configPath) throw new Error('Pass an explicit diagnostic config JSON');
const configBytes = await fs.readFile(configPath), config = JSON.parse(configBytes);
if (config.diagnosticOnly !== true) throw new Error('Full TJS runner currently supports diagnostic work only');
const here = path.dirname(fileURLToPath(import.meta.url));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function fileHash(file) {
  const checksum = createHash('sha256');
  for await (const chunk of createReadStream(file)) checksum.update(chunk);
  return checksum.digest('hex');
}
const modelReceiptPath = path.join(config.modelDir, 'receipt.json');
const cpuReceiptPath = path.join(config.cpuDir, 'receipt.json');
const modelReceipt = JSON.parse(await fs.readFile(modelReceiptPath));
const cpuReceipt = JSON.parse(await fs.readFile(cpuReceiptPath));
const modelConfig = JSON.parse(await fs.readFile(path.join(config.modelDir, 'config.json')));
const basePolicy = JSON.parse(await fs.readFile(path.join(here, 'contract.json')));
if (!modelReceipt.completed || !cpuReceipt.completed ||
    Object.keys(modelReceipt.inputs).some(key => modelReceipt.inputs[key] !== cpuReceipt.inputs[key]) ||
    modelReceipt.contract.kvPhysicalDtype !== cpuReceipt.contract.kvPhysicalDtype) {
  throw new Error('Frozen ONNX/CPU canonical identity or physical-cache lane differs');
}
const initializer = JSON.parse(await fs.readFile(modelReceipt.initializerBinding.path));
if (!initializer.pass || initializer.parameterCount !== 321 ||
    await fileHash(modelReceipt.initializerBinding.path) !== modelReceipt.initializerBinding.sha256) {
  throw new Error('Complete canonical initializer proof is missing or changed');
}
const policy = { ...basePolicy, textExport: modelReceipt.contract };
const externalFiles = new Map();
for (const artifact of modelReceipt.artifacts) {
  const file = path.resolve(config.modelDir, artifact.path);
  if (await fileHash(file) !== artifact.sha256) throw new Error(`Frozen ONNX artifact changed: ${file}`);
  if (file.endsWith('.data')) externalFiles.set(path.basename(file), file);
}
const sessionFiles = modelReceipt.sessionFiles;
if (!sessionFiles) throw new Error('Explicit session paths are required by the diagnostic TJS adapter');
const sessionDescriptions = {};
const nameMap = { embedding: 'embed_tokens', prefill: 'decoder_prefill', decode: 'decoder_decode' };
for (const [name, graph] of Object.entries(nameMap)) {
  const filenames = new Set(initializer.graphs[graph].map(entry => entry.externalData?.location).filter(Boolean));
  sessionDescriptions[name] = { url: `/graph/${name}.onnx`, externalData: [...filenames].map(filename => {
    if (!externalFiles.has(filename)) throw new Error(`External initializer file is not frozen: ${filename}`);
    return { path: filename, url: `/external/${encodeURIComponent(filename)}` };
  }) };
}
const nativeFiles = new Map(cpuReceipt.cases.flatMap(test => test.logits.filter(row => row.engine === 'native')
  .map(row => [row.file, path.join(config.cpuDir, row.file)])));
await fs.mkdir(config.outputDir, { recursive: false });
const identities = { configSha256: sha(configBytes), modelReceipt: { path: modelReceiptPath, sha256: await fileHash(modelReceiptPath) },
  cpuReceipt: { path: cpuReceiptPath, sha256: await fileHash(cpuReceiptPath) }, files: {} };
for (const file of ['run_text_gpu.js', 'text_gpu_browser.js', 'tjs_model.js']) identities.files[file] = await fileHash(path.join(here, file));
for (const [name, file] of [['tjs', path.join(config.tjsDist, 'transformers.web.min.js')],
  ['ort', path.join(config.ortDist, 'ort.webgpu.min.mjs')],
  ['ortWasm', path.join(config.ortDist, 'ort-wasm-simd-threaded.jsep.wasm')]]) {
  identities.files[name] = { path: file, sha256: await fileHash(file) };
}
await fs.writeFile(path.join(config.outputDir, 'identity.json'), JSON.stringify(identities, null, 2));
const descriptor = { config, modelReceipt, cpuReceipt, modelConfig, policy, sessions: sessionDescriptions,
  cpuGate: { path: cpuReceiptPath, sha256: await fileHash(cpuReceiptPath), pass: cpuReceipt.pass,
    failedChecks: cpuReceipt.checks.filter(check => !check.pass).length, qualification: false },
  sessionInitializerBytes: Object.fromEntries(Object.entries(initializer.graphs).map(([name, entries]) =>
    [name, entries.reduce((sum, entry) => sum + entry.bytes, 0)])) };
const logs = [], errors = [], events = [];
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/event' && request.method === 'POST') {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const event = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      events.push({ ...event, observedAt: new Date().toISOString() });
      await fs.writeFile(path.join(config.outputDir, 'progress.json'), JSON.stringify(events, null, 2));
      console.log(JSON.stringify(event)); response.end('ok'); return;
    }
    if (pathname === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html' });
      response.end('<script type="module" src="/adapter/text_gpu_browser.js"></script>'); return;
    }
    if (pathname === '/probe.json') {
      response.writeHead(200, { 'Content-Type': 'application/json' }); response.end(JSON.stringify(descriptor)); return;
    }
    let file;
    const graph = pathname.match(/^\/graph\/(embedding|prefill|decode)\.onnx$/)?.[1];
    if (graph) file = sessionFiles[nameMap[graph]];
    else if (pathname.startsWith('/external/')) file = externalFiles.get(decodeURIComponent(pathname.slice('/external/'.length)));
    else if (pathname.startsWith('/native/')) file = nativeFiles.get(decodeURIComponent(pathname.slice('/native/'.length)));
    else for (const [prefix, root] of [['/adapter/', here], ['/ort/', config.ortDist], ['/tjs/', config.tjsDist]]) {
      if (pathname.startsWith(prefix)) {
        file = path.resolve(root, decodeURIComponent(pathname.slice(prefix.length)));
        if (!file.startsWith(path.resolve(root) + path.sep)) throw new Error('Path outside served root');
        break;
      }
    }
    if (!file) { response.writeHead(404); response.end(); return; }
    const stat = await fs.stat(file), extension = path.extname(file);
    const type = ['.js', '.mjs'].includes(extension) ? 'text/javascript' : extension === '.wasm' ? 'application/wasm' : 'application/octet-stream';
    response.writeHead(200, { 'Content-Type': type, 'Content-Length': stat.size });
    createReadStream(file).on('error', error => response.destroy(error)).pipe(response);
  } catch (error) { if (!response.headersSent) response.writeHead(500); response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let context;
try {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, 'browser-profile'), {
    executablePath: config.browserExecutable, args: config.browserArgs, headless: true });
  const page = await context.newPage();
  page.on('console', message => logs.push(`${message.type()}: ${message.text()}`));
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => globalThis.RESULT !== undefined, null, { timeout: config.timeoutMs });
  const result = await page.evaluate(() => globalThis.RESULT);
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify({ ...result, browserErrors: errors }, null, 2));
  if (result.error || errors.length) throw new Error(result.error || errors.join('\n'));
} finally {
  await context?.close(); await new Promise(resolve => server.close(resolve));
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
