import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright';
import { createProbeServer, probeOrigin, closeProbeServer } from './probe-server.js';
import { preparePooledRounding } from './prepare-pooled.js';

const configBytes = await fs.readFile(process.argv[2], 'utf8');
const config = JSON.parse(configBytes);
const browserEntry = config.browserEntry ?? 'probe-browser.js';
if (!/^[a-z0-9-]+\.js$/.test(browserEntry)) throw new Error('Invalid browser observation entrypoint');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const here = path.dirname(fileURLToPath(import.meta.url));
await fs.mkdir(config.outputDir, { recursive: false });
await fs.cp(path.join(config.pooledRoot, 'engine'), path.join(config.outputDir, 'vendor', 'engine'), { recursive: true, dereference: true });
const vendorChanges = await preparePooledRounding(path.join(config.outputDir, 'vendor'), config.pooledRoundingPreparation);
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
await fs.writeFile(path.join(config.outputDir, 'source-identity.json'), JSON.stringify({ configSha256: sha(configBytes), contractSha256: sha(contractBytes), vendorChanges, engineFiles: identity,
  nativeKvShaderSha256: config.nativeKvShader ? sha(await fs.readFile(config.nativeKvShader)) : null,
  adapterFiles: await Promise.all(
  ['run-pooled-probe.js', 'probe-browser.js', 'pooled-canonical.js', 'verify-pooled-gpu.js', 'shared-input.js', 'capture-pooled-boundaries.js',
    'rne-probe-browser.js', 'kv-cache-rne-prototype.wgsl', 'prepare-pooled.js', 'timing.js', 'probe-server.js', 'pooled-session.js'].map(async name => ({ name, sha256: sha(await fs.readFile(path.join(here, name))) }))) }, null, 2));
const logs = [], errors = [];
const server = await createProbeServer({ entry: browserEntry, config: { ...config, contract },
  routes: [['/vendor/', path.join(config.outputDir, 'vendor')], ['/canonical/', config.canonicalDir], ['/adapter/', here]],
  files: { '/model.gguf': config.ggufPath, '/native-kv.wgsl': config.nativeKvShader },
});
let context;
try {
  context = await chromium.launchPersistentContext(path.join(config.outputDir, 'browser-profile'), {
    executablePath: config.browserExecutable, args: config.browserArgs, headless: true });
  const page = await context.newPage();
  page.on('console', message => { logs.push(`${message.type()}: ${message.text()}`); console.log(message.text()); });
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${probeOrigin(server)}/`);
  await page.waitForFunction(() => globalThis.RESULT !== undefined, null, { timeout: config.timeoutMs });
  const result = await page.evaluate(() => globalThis.RESULT);
  await fs.writeFile(path.join(config.outputDir, 'result.json'), JSON.stringify({ ...result, browserErrors: errors }, null, 2));
  if (result.error || errors.length || result.pass === false) throw new Error(result.error || errors.join('\n') || 'Observation acceptance failed');
} finally {
  await context?.close(); await closeProbeServer(server);
  await fs.writeFile(path.join(config.outputDir, 'browser.log'), logs.join('\n'));
}
