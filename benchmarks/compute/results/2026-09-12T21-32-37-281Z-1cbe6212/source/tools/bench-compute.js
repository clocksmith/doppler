import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import { RECEIPT_SCHEMA, requireCondition, validateSuite } from '../benchmarks/compute/contract.js';
import { analyzeComparison } from '../benchmarks/compute/analysis.js';
import { compareOutput, referenceLinear } from '../benchmarks/compute/linear.js';
import { renderReport } from '../benchmarks/compute/report.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const sourcePaths = [
  'tools/bench-compute.js', 'package.json', 'benchmarks/benchmark-schema.json',
  'benchmarks/compute/contract.js', 'benchmarks/compute/linear.js', 'benchmarks/compute/linear.wgsl',
  'benchmarks/compute/override_probe.wgsl', 'benchmarks/compute/executor.js', 'benchmarks/compute/analysis.js',
  'benchmarks/compute/runner.js', 'benchmarks/compute/report.js', 'benchmarks/compute/index.html',
  'benchmarks/compute/README.md', 'src/debug/stats.js', 'src/gpu/kernels/bias_add.wgsl', 'src/gpu/kernels/silu.wgsl',
  'src/tooling/node-browser/static-server.js',
];
const digest = bytes => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 2) {
  requireCondition(['--config', '--out', '--analyze'].includes(args[i]) && args[i + 1] && !Object.hasOwn(options, args[i]), 'Usage: node tools/bench-compute.js [--config suite.json] [--out NEW_DIRECTORY] | --analyze receipt.json');
  options[args[i]] = args[i + 1];
}

function within(base, path) {
  const full = resolve(base, path);
  const rel = relative(base, full);
  requireCondition(rel && rel !== '..' && !rel.startsWith(`..${sep}`) && !rel.startsWith(sep), 'Artifact path escapes its run directory.');
  return full;
}

if (options['--analyze']) {
  requireCondition(Object.keys(options).length === 1, '--analyze cannot change the measured config or output directory.');
  const receiptPath = resolve(options['--analyze']);
  const out = dirname(receiptPath);
  const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
  requireCondition(receipt.schema === RECEIPT_SCHEMA && receipt.quality.passed, 'Cannot qualify an incomplete or failed receipt.');
  validateSuite(receipt.config);
  for (const [path, hash] of Object.entries(receipt.provenance.sources)) {
    requireCondition(digest(await readFile(within(resolve(out, 'source'), path))) === hash, `Source identity mismatch: ${path}`);
  }
  const binary = new Map();
  for (const artifact of receipt.artifacts) {
    const bytes = await readFile(within(out, artifact.path));
    requireCondition(bytes.length === artifact.bytes && digest(bytes) === artifact.sha256, `Artifact identity mismatch: ${artifact.path}`);
    requireCondition(bytes.length % 4 === 0, 'Float artifact has an invalid byte length.');
    binary.set(artifact.path, new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
  }
  requireCondition(receipt.metrics.comparisons.length === receipt.config.cases.length * receipt.config.experiments.length, 'Missing comparison rows.');
  for (let i = 0; i < receipt.metrics.comparisons.length; i += 1) {
    const row = receipt.metrics.comparisons[i];
    const data = Object.fromEntries(['x', 'weights', 'bias'].map(key => [key, binary.get(row.inputArtifacts[key].path)]));
    const expected = referenceLinear(row.shape, data, receipt.config.shared);
    requireCondition(digest(new Uint8Array(expected.buffer)) === row.inputArtifacts.expected.sha256, 'Saved oracle differs from independently recomputed arithmetic.');
    for (const role of ['baseline', 'candidate']) requireCondition(compareOutput(binary.get(row.outputs[role].path), expected, receipt.config.shared).passed, 'Saved output failed independent reanalysis.');
    row.summary = analyzeComparison(row, receipt.config, i);
  }
  await writeFile(resolve(out, 'reanalyzed-report.html'), renderReport(receipt));
  console.log(JSON.stringify({ status: 'reanalyzed', comparisons: receipt.metrics.comparisons.length, report: resolve(out, 'reanalyzed-report.html') }, null, 2));
} else {
  requireCondition(os.endianness() === 'LE', 'This artifact writer currently requires a little-endian host.');
  const configPath = resolve(options['--config'] ?? resolve(root, 'benchmarks/compute/suite.json'));
  const configBytes = await readFile(configPath);
  const suite = validateSuite(JSON.parse(configBytes));
  const runId = `${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}`;
  const out = resolve(options['--out'] ?? resolve(root, 'benchmarks/compute/results', runId));
  await mkdir(dirname(out), { recursive: true });
  await mkdir(out);
  const sourceRoot = resolve(out, 'source');
  const sources = {};
  for (const path of sourcePaths) {
    const bytes = await readFile(resolve(root, path));
    const destination = within(sourceRoot, path);
    await mkdir(dirname(destination), { recursive: true });
    await writeFile(destination, bytes);
    sources[path] = digest(bytes);
  }
  await writeFile(resolve(sourceRoot, 'benchmarks/compute/suite.json'), configBytes);
  sources['benchmarks/compute/suite.json'] = digest(configBytes);
  const provenance = { sourceIdentity: digest(JSON.stringify(sources)), sources,
    command: `node tools/bench-compute.js --config ${JSON.stringify(configPath)} --out ${JSON.stringify(out)}`,
    replay: 'From the source snapshot, run node tools/bench-compute.js with a new --out directory and the same browser installation. No model downloads or publication are performed.',
    host: { platform: process.platform, arch: process.arch, osRelease: os.release(), cpuModel: os.cpus()[0]?.model ?? null, node: process.version },
    browserLaunch: { channel: suite.engine.channel, headless: suite.engine.headless, args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features'] } };
  let receipt = { schema: RECEIPT_SCHEMA, schemaVersion: 1, timestamp: new Date().toISOString(), suite: suite.id, runType: 'paired-operator-ablation', env: { ...provenance.host }, model: { kind: 'none' }, config: suite, workload: { cases: suite.cases }, metrics: { comparisons: [] }, quality: { passed: false, productionPromotionAllowed: false }, failures: [], artifacts: [], provenance };
  await writeFile(resolve(out, 'receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`);
  let server;
  let browser;
  let timer;
  const artifacts = [];
  try {
    const { chromium } = await import('playwright');
    const { createStaticFileServer } = await import('../src/tooling/node-browser/static-server.js');
    server = await createStaticFileServer({ rootDir: sourceRoot, host: '127.0.0.1', port: 0 });
    browser = await chromium.launch({ ...provenance.browserLaunch, timeout: suite.shared.timeoutMs });
    const browserVersion = browser.version();
    const session = await browser.newBrowserCDPSession();
    let systemGpu;
    try { systemGpu = (await session.send('SystemInfo.getInfo')).gpu; } finally { await session.detach(); }
    const page = await browser.newPage();
    const consoleMessages = [];
    page.on('console', message => { if (message.type() === 'error' || message.type() === 'warning') consoleMessages.push({ type: message.type(), text: message.text() }); });
    page.on('pageerror', error => consoleMessages.push({ type: 'pageerror', text: error.message }));
    await page.exposeFunction('persistComputeArtifact', async (path, values) => {
      requireCondition(/^data\/[a-z0-9/-]+\.bin$/.test(path), 'Unexpected artifact path.');
      const bytes = Buffer.from(values);
      const destination = within(out, path);
      await mkdir(dirname(destination), { recursive: true });
      await writeFile(destination, bytes, { flag: 'wx' });
      const artifact = { path, bytes: bytes.length, sha256: digest(bytes), encoding: 'float32-le' };
      artifacts.push(artifact);
      return artifact;
    });
    await page.exposeFunction('computeProgress', value => {
      console.log(JSON.stringify(value));
      return writeFile(resolve(out, 'progress.json'), `${JSON.stringify(value, null, 2)}\n`);
    });
    await page.goto(`${server.baseUrl}/benchmarks/compute/index.html`, { waitUntil: 'load', timeout: suite.shared.timeoutMs });
    const execution = page.evaluate(async payload => {
      const { runComputeEvidence } = await import('/benchmarks/compute/runner.js');
      return runComputeEvidence(payload, {
        progress: value => globalThis.computeProgress(value),
        artifact: (path, array) => globalThis.persistComputeArtifact(path, Array.from(new Uint8Array(array.buffer, array.byteOffset, array.byteLength))),
      });
    }, suite);
    const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`Compute suite exceeded its declared ${suite.shared.timeoutMs} ms budget.`)), suite.shared.timeoutMs); });
    receipt = await Promise.race([execution, timeout]);
    receipt.provenance = provenance;
    Object.assign(receipt.env, { browserVersion, host: provenance.host, systemGpu, consoleMessages });
  } catch (error) {
    receipt.quality.passed = false;
    receipt.failures.push({ name: error.name, message: error.message, stack: error.stack });
    receipt.artifacts = artifacts;
  } finally {
    clearTimeout(timer);
    await browser?.close().catch(error => receipt.failures.push({ phase: 'browser-cleanup', message: error.message }));
    await server?.close().catch(error => receipt.failures.push({ phase: 'server-cleanup', message: error.message }));
    if (receipt.failures.length) receipt.quality.passed = false;
    await writeFile(resolve(out, 'receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`);
    await writeFile(resolve(out, 'report.html'), renderReport(receipt));
  }
  console.log(JSON.stringify({ status: receipt.quality.passed ? 'passed' : 'failed', comparisons: receipt.metrics.comparisons.length, failures: receipt.failures, receipt: resolve(out, 'receipt.json'), report: resolve(out, 'report.html') }, null, 2));
  if (!receipt.quality.passed) process.exitCode = 1;
}
