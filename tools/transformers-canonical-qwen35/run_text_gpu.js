import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { cachePorts } from './tjs_model.js';

const configPath = process.argv[2];
if (!configPath) throw new Error('Pass an explicit diagnostic config JSON');
const configBytes = await fs.readFile(configPath), config = JSON.parse(configBytes);
if (config.diagnosticOnly !== true) throw new Error('Full TJS runner currently supports diagnostic work only');
if (!['short-cpu', 'frozen-teacher'].includes(config.numericalMode)) throw new Error('Select explicit numericalMode: short-cpu or frozen-teacher');
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
const teacherPath = config.numericalMode === 'frozen-teacher' ? path.join(config.teacherDir, 'receipt.json') : null;
const teacher = teacherPath ? JSON.parse(await fs.readFile(teacherPath)) : null;
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
if (teacher && (!teacher.completed || teacher.contract.maxNewTokens !== policy.teacher.maxNewTokens ||
    JSON.stringify(teacher.contract.eosTokenIds) !== JSON.stringify(policy.teacher.eosTokenIds) ||
    teacher.contract.kvPhysicalDtype !== modelReceipt.contract.kvPhysicalDtype ||
    JSON.stringify(teacher.inputs) !== JSON.stringify(modelReceipt.inputs) || teacher.prompts.length !== 4)) {
  throw new Error('Frozen four-prompt teacher identity, budget, EOS or physical-cache lane differs');
}
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
if (modelReceipt.contract.prefillBuckets) {
  delete nameMap.prefill;
  for (const bound of modelReceipt.contract.prefillBuckets) nameMap['prefill_' + bound] = 'decoder_prefill_' + bound;
}
for (const [name, graph] of Object.entries(nameMap)) {
  const filenames = new Set(initializer.graphs[graph].map(entry => entry.externalData?.location).filter(Boolean));
  sessionDescriptions[name] = { graphName: graph, url: `/graph/${name}.onnx`, externalData: [...filenames].map(filename => {
    if (!externalFiles.has(filename)) throw new Error(`External initializer file is not frozen: ${filename}`);
    return { path: filename, url: `/external/${encodeURIComponent(filename)}` };
  }) };
}
const nativeFiles = new Map(cpuReceipt.cases.flatMap(test => test.logits.filter(row => row.engine === 'native')
  .map(row => [row.file, path.join(config.cpuDir, row.file)])));
if (teacher) for (const prompt of teacher.prompts) for (const row of prompt.snapshots) {
  const file = path.join(config.teacherDir, row.file);
  if (await fileHash(file) !== row.sha256) throw new Error(`Frozen teacher snapshot differs: ${file}`);
  nativeFiles.set(row.file, file);
}
await fs.mkdir(config.outputDir, { recursive: false });
const identities = { configSha256: sha(configBytes), modelReceipt: { path: modelReceiptPath, sha256: await fileHash(modelReceiptPath) },
  cpuReceipt: { path: cpuReceiptPath, sha256: await fileHash(cpuReceiptPath) }, files: {} };
for (const file of ['run_text_gpu.js', 'text_gpu_browser.js', 'tjs_model.js', 'bucketed_prefill.js', 'teacher_cases.js', 'observed_session.js', 'observe_gpu.js']) identities.files[file] = await fileHash(path.join(here, file));
if (teacher) identities.teacher = { path: teacherPath, sha256: await fileHash(teacherPath) };
for (const [name, file] of [['tjs', path.join(config.tjsDist, 'transformers.web.min.js')],
  ['ort', path.join(config.ortDist, 'ort.webgpu.min.mjs')],
  ['ortWasm', path.join(config.ortDist, 'ort-wasm-simd-threaded.jsep.wasm')]]) {
  identities.files[name] = { path: file, sha256: await fileHash(file) };
}
await fs.writeFile(path.join(config.outputDir, 'identity.json'), JSON.stringify(identities, null, 2));
const descriptor = { config, modelReceipt, cpuReceipt, teacher, modelConfig, policy, sessions: sessionDescriptions,
  cpuGate: { path: cpuReceiptPath, sha256: await fileHash(cpuReceiptPath), pass: cpuReceipt.pass,
    failedChecks: cpuReceipt.checks.filter(check => !check.pass).length, qualification: false },
  sessionInitializerBytes: Object.fromEntries(Object.entries(initializer.graphs).map(([name, entries]) =>
    [name, entries.reduce((sum, entry) => sum + entry.bytes, 0)])) };
const logs = [], errors = [], events = [];
const observationExpected = new Map(Object.entries(nameMap).flatMap(([session, graph]) =>
  (modelReceipt.readbackPorts?.[graph] ?? []).map(port => [`${session}:${port.output}`, { ...port, session }])));
const observations = new Map();
const stateObservations = new Map();
if ((observationExpected.size || config.captureStateSnapshots === true) && (!Number.isSafeInteger(config.weightReadbackChunkBytes) ||
    config.weightReadbackChunkBytes < 4 || config.weightReadbackChunkBytes % 4)) {
  throw new Error('GPU observation requires explicit positive, four-byte-aligned weightReadbackChunkBytes');
}
const server = http.createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname.startsWith('/state-readback/') && request.method === 'POST') {
      if (config.captureStateSnapshots !== true || !teacher) throw new Error('State capture is not enabled for a frozen teacher');
      const url = new URL(request.url, 'http://localhost');
      const key = decodeURIComponent(pathname.slice('/state-readback/'.length));
      const [promptId, rawStep, name] = key.split(':');
      const step = Number(rawStep), offset = Number(url.searchParams.get('offset'));
      const prompt = teacher.prompts.find(value => value.id === promptId);
      const ports = cachePorts(modelConfig, modelReceipt.contract.kvPhysicalDtype);
      const index = ports.findIndex(value => value.output === name), port = ports[index];
      if (!prompt || !teacher.contract.snapshotSteps.includes(step) || !port || port.type !== 'float32' ||
          !Number.isSafeInteger(offset) || offset < 0) throw new Error('State observation is outside the frozen F32 snapshot contract');
      const shape = port.shape.map(value => value ?? prompt.promptIds.length + step);
      const bytes = 4 * shape.reduce((a, b) => a * b, 1);
      let progress = stateObservations.get(key);
      if (!progress) {
        const file = `gpu-state-${promptId}-${step}-${index}.bin`;
        progress = { offset: 0, checksum: createHash('sha256'), receipt: null,
          file, handle: await fs.open(path.join(config.outputDir, file), 'wx') };
        stateObservations.set(key, progress);
      }
      if (progress.receipt || progress.offset !== offset) throw new Error('State ranges must be complete and sequential');
      let received = 0;
      for await (const chunk of request) {
        if (received + chunk.length > config.weightReadbackChunkBytes || offset + received + chunk.length > bytes) {
          throw new Error('State observation exceeds declared range');
        }
        await progress.handle.write(chunk, 0, chunk.length, offset + received);
        received += chunk.length;
        progress.checksum.update(chunk);
      }
      if (!received || received % 4) throw new Error('State observation is empty or misaligned');
      progress.offset += received;
      if (progress.offset === bytes) {
        await progress.handle.close(); progress.handle = null;
        progress.receipt = { promptId, step, name, file: progress.file, shape, bytes, dtype: 'float32',
          sha256: progress.checksum.digest('hex'), scope: 'Actual GPU state bytes; numerical comparison pending' };
      }
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ bytes: progress.offset, receipt: progress.receipt })); return;
    }
    if (pathname.startsWith('/weight-readback/') && request.method === 'POST') {
      const url = new URL(request.url, 'http://localhost');
      const key = decodeURIComponent(pathname.slice('/weight-readback/'.length));
      const expected = observationExpected.get(key), offset = Number(url.searchParams.get('offset'));
      if (!expected || !Number.isSafeInteger(offset) || offset < 0) throw new Error('Unknown or invalid GPU observation range');
      let progress = observations.get(key);
      if (!progress) { progress = { offset: 0, checksum: createHash('sha256'), receipt: null }; observations.set(key, progress); }
      if (progress.receipt || progress.offset !== offset) throw new Error('GPU observation ranges must be complete and sequential');
      let received = 0;
      for await (const chunk of request) {
        received += chunk.length;
        if (received > config.weightReadbackChunkBytes || offset + received > expected.bytes) throw new Error('GPU observation exceeds declared range');
        progress.checksum.update(chunk);
      }
      if (!received || received % 4) throw new Error('GPU observation is empty or misaligned');
      progress.offset += received;
      if (progress.offset === expected.bytes) {
        const observedSha256 = progress.checksum.digest('hex');
        progress.receipt = { ...expected, observedSha256, pass: observedSha256 === expected.expectedSha256,
          transport: 'Actual inference-session GPU Identity output, chunked copyBufferToBuffer/mapAsync readback' };
      }
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ bytes: progress.offset, receipt: progress.receipt })); return;
    }
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
      response.end('<script type="importmap">{"imports":{"onnxruntime-web/webgpu":"/ort/ort.webgpu.min.mjs",' +
        '"onnxruntime-common":"/ort/ort.webgpu.min.mjs"}}</script>' +
        '<script type="module" src="/adapter/text_gpu_browser.js"></script>'); return;
    }
    if (pathname === '/probe.json') {
      response.writeHead(200, { 'Content-Type': 'application/json' }); response.end(JSON.stringify(descriptor)); return;
    }
    let file;
    const graph = pathname.match(/^\/graph\/([a-z0-9_]+)\.onnx$/)?.[1];
    if (graph && nameMap[graph]) file = sessionFiles[nameMap[graph]];
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
  result.materializationObservation = { requiredBindings: observationExpected.size,
    completedBindings: [...observations.values()].filter(value => value.receipt).length,
    checks: [...observations.values()].flatMap(value => value.receipt ? [value.receipt] : []),
    pass: observationExpected.size > 0 && observations.size === observationExpected.size &&
      [...observations.values()].every(value => value.receipt?.pass) };
  result.materializationObservation.canonicalTensors = new Set(result.materializationObservation.checks.map(check => check.source)).size;
  result.stateObservations = [...stateObservations.values()].flatMap(value => value.receipt ? [value.receipt] : []);
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify({ ...result, browserErrors: errors }, null, 2));
  if (result.error || errors.length) throw new Error(result.error || errors.join('\n'));
} finally {
  await context?.close(); await new Promise(resolve => server.close(resolve));
  for (const value of stateObservations.values()) await value.handle?.close();
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
