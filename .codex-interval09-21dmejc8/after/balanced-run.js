import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { retainRequestObservation } from './request-tail.js';
import { captureCpuProfile } from './post-acceptance-cpu-profile.js';
import { createProbeServer, probeOrigin, closeProbeServer } from '../pooled-equal-weights/probe-server.js';
import { same, validatePair, validateVariantIntake, validateResolvedInference, validateDistinctResolvedVariants,
  acceptanceContract, requestMetrics, runWorkload, validateComparisonMode } from './balanced-protocol.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const input = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
if (input.postAcceptanceCpuProfile !== undefined && typeof input.postAcceptanceCpuProfile !== 'boolean') {
  throw Error('postAcceptanceCpuProfile must be an explicit boolean');
}
if (!Array.isArray(input.variants) || input.variants.length !== 2) throw Error('Exactly two variants required');
const variants = await Promise.all(input.variants.map(async entry => {
  if (!/^[a-z][a-z0-9-]*$/.test(entry.id)) throw Error('Unsafe variant id');
  const configBytes = await fs.readFile(entry.configPath);
  const config = JSON.parse(configBytes);
  const contract = JSON.parse(await fs.readFile(config.contractPath));
  const runtimeRoot = await fs.realpath(config.runtimeRoots?.doppler ?? config.dopplerRoot);
  return { ...entry, config, configSha256: sha(configBytes), contract, runtimeRoot };
}));
for (const variant of variants) variant.configPath = await fs.realpath(variant.configPath);
validateComparisonMode(input.comparisonMode);
validateVariantIntake(variants);
validatePair(variants.map(value => value.config), variants.map(value => value.contract), input.comparisonMode);
const contract = acceptanceContract(variants[0].contract);
const output = path.resolve(input.outputDir);
await fs.mkdir(output, { recursive: false });
const save = (name, value) => fs.writeFile(path.join(output, name), JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
const receipt = { startedAt: new Date().toISOString(), complete: false, attempts: [], gates: {}, cleanupErrors: [] };
const writeReceipt = () => fs.writeFile(path.join(output, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
await fs.writeFile(path.join(output, 'boundaries.jsonl'), '', { flag: 'wx' });
const phase = async (label, details = {}) => {
  const state = { atUtc: new Date().toISOString(), ...details, phase: label };
  if (label === 'quality-passed') receipt.qualityPassed = true;
  console.log(JSON.stringify(state));
  await fs.appendFile(path.join(output, 'boundaries.jsonl'), JSON.stringify(state) + '\n');
  await fs.writeFile(path.join(output, 'live-phase.json'), JSON.stringify(state, null, 2) + '\n');
};
const snapshot = path.join(output, 'executed-source');
for (const rel of ['qwen-economics/browser.js', 'pooled-equal-weights/shared-input.js',
  'pooled-equal-weights/probe-server.js', 'qwen-economics/compare-reference-moe.py',
  'qwen-economics/normalize-reference-manifest.js',
  'qwen-economics/balanced-run.js', 'qwen-economics/balanced-protocol.js',
  'qwen-economics/post-acceptance-cpu-profile.js', 'qwen-economics/request-tail.js']) {
  const target = path.join(snapshot, rel);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.copyFile(path.join(path.dirname(here), rel), target, 1);
}
const snapshotHashes = {};
for (const rel of ['qwen-economics/browser.js', 'qwen-economics/compare-reference-moe.py',
  'qwen-economics/normalize-reference-manifest.js',
  'qwen-economics/balanced-run.js', 'qwen-economics/balanced-protocol.js',
  'qwen-economics/post-acceptance-cpu-profile.js', 'qwen-economics/request-tail.js']) {
  snapshotHashes[rel] = sha(await fs.readFile(path.join(snapshot, rel)));
}
const manifestBytes = await fs.readFile(path.join(variants[0].config.rdrrDir, 'manifest.json'));
const weightIdentity = { realpath: await fs.realpath(variants[0].config.rdrrDir), manifestSha256: sha(manifestBytes),
  note: 'Both servers resolve the same immutable model path; manifest hashes identify artifact shards. No costly whole-model rehash.' };
const owned = [];
let timedOut = false;
async function bounded(page, fn, arg, limit) {
  let timer;
  return Promise.race([page.evaluate(fn, arg), new Promise((_, reject) => {
    timer = setTimeout(() => { timedOut = true; reject(Error('Request exceeded requestTimeoutMs ' + limit)); }, limit);
  })]).finally(() => clearTimeout(timer));
}
async function sourceHashes(root) {
  const hashes = {};
  async function walk(directory) {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile()) hashes[path.relative(root, full)] = sha(await fs.readFile(full));
    }
  }
  await walk(path.join(root, 'src'));
  return hashes;
}
async function reference(item) {
  const probe = item.config.referenceProbe;
  const directory = path.join(output, item.id, 'reference-probe');
  await fs.mkdir(directory, { recursive: false });
  const saveProbe = (name, value) => save(path.join(item.id, 'reference-probe', name), value);
  await saveProbe('contract.json', probe.contract);
  await saveProbe('execution.json', { ...item.config, phase: 'reference-probe',
    contractSha256: sha(JSON.stringify(probe.contract)) });
  await saveProbe('doppler-ready.json', item.ready);
  const task = probe.contract.shared.qualityTasks.find(value => value.id === 'arithmetic');
  if (!task) throw Error('Reference arithmetic task missing');
  await phase('reference-start', { variant: item.id });
  let result;
  try {
    result = await bounded(item.page, task => globalThis.REQUEST(task, 'reference-probe'), task, item.config.requestTimeoutMs);
  } catch (error) { result = { error: String(error.stack || error), controllerFailure: true }; }
  await saveProbe('arithmetic-0-doppler.json', result);
  await saveProbe('receipt.json', { complete: !result.error, scope: 'Completed reference phase only; workload remains pending',
    attempts: [{ kind: 'doppler', task: 'arithmetic', repeat: 0, file: 'arithmetic-0-doppler.json', error: result.error ?? null }] });
  if (result.error) throw Error(result.error);
  const gatePath = path.join(directory, 'numerical-gate.json');
  try {
    const stdout = execFileSync('python3', [path.join(snapshot, 'qwen-economics/compare-reference-moe.py'),
      '--decode-steps', '1', '--reference', probe.referenceDir, '--gpu', directory,
      '--contract', probe.referenceContractPath, '--model', item.config.rdrrDir,
      '--runtime', item.runtimeRoot, '--output', gatePath], { stdio: 'pipe' });
    await fs.writeFile(path.join(directory, 'comparator-stdout.log'), stdout, { flag: 'wx' });
  } catch (error) {
    await fs.writeFile(path.join(directory, 'comparator-error.log'),
      String(error.stack || error) + '\n' + String(error.stdout ?? '') + '\n' + String(error.stderr ?? ''), { flag: 'wx' });
    throw error;
  }
  const gate = JSON.parse(await fs.readFile(gatePath));
  if (gate.passed !== true) throw Error(item.id + ' reference numerical gate rejected');
  receipt.gates[item.id] = { numericalPassed: true, gate: path.relative(output, gatePath) };
  await writeReceipt();
  await phase('reference-passed', { variant: item.id, maxAbs: gate.checks.fullVocabulary.maxAbs });
}
try {
  await save('contract.json', contract);
  await save('execution.json', { ...input, protocol: { warmupPairs: 1, timedPairsPerTask: 10,
    pairOrders: 'AB,BA repeated five times; warmup AB excluded', maxNewTokens: 128 }, snapshotHashes, weightIdentity });
  for (const variant of variants) {
    await fs.mkdir(path.join(output, variant.id));
    await save(path.join(variant.id, 'input-config.json'), variant.config);
    const config = { ...variant.config, contract, kind: 'doppler', variantId: variant.id,
      configPath: variant.configPath, countGpuApi: false,
      sampleResources: false, diagnosticTiming: false, profileGeneration: false };
    await save(path.join(variant.id, 'execution.json'), { ...config, configPath: variant.configPath,
      configSha256: variant.configSha256, runtimeRoot: variant.runtimeRoot, snapshotHashes });
    variant.sourceHashes = await sourceHashes(variant.runtimeRoot);
    await save(path.join(variant.id, 'runtime-source-sha256.json'), variant.sourceHashes);
    await phase('load-start', { variant: variant.id });
    const server = await createProbeServer({ entry: 'qwen-economics/browser.js', config,
      routes: [['/adapter/', snapshot], ['/runtime/', variant.runtimeRoot], ['/pooled/', config.pooledRoot], ['/rdrr/', config.rdrrDir]],
      files: { '/model.gguf': config.ggufPath } });
    const item = { ...variant, config, server, context: null, page: null, logs: [] };
    owned.push(item);
    item.context = await chromium.launchPersistentContext(path.join(config.profileRoot, path.basename(output), item.id), {
      headless: true, executablePath: config.browserExecutable, args: config.browserArgs });
    item.page = await item.context.newPage();
    for (const event of ['console', 'pageerror']) item.page.on(event, value => {
      const line = new Date().toISOString() + ' ' + event + ': ' + (event === 'console' ? value.text() : String(value));
      item.logs.push(line);
      fs.appendFile(path.join(output, item.id, 'live.log'), line + '\n').catch(() => {});
    });
    let rejectPageError;
    const pageFailure = new Promise((_, reject) => { rejectPageError = reject; });
    item.page.on('pageerror', error => rejectPageError(Error(item.id + ' page error before READY: ' + String(error))));
    await item.page.goto(probeOrigin(server));
    await Promise.race([item.page.waitForFunction(() => globalThis.READY !== undefined, null,
      { timeout: config.loadTimeoutMs }), pageFailure]);
    item.ready = await item.page.evaluate(() => globalThis.READY);
    await save(path.join(item.id, 'doppler-ready.json'), item.ready);
    if (item.ready.error) throw Error(item.id + ': ' + item.ready.error);
    await phase('load-ready', { variant: item.id, loadMs: item.ready.loadMs });
  }
  for (const key of ['kind', 'adapter', 'features', 'manifest', 'userAgent', 'observedWeights']) {
    same(owned[0].ready[key], owned[1].ready[key], 'Actual READY identity mismatch: ' + key);
  }
  validateResolvedInference(owned.map(item => item.ready.resolvedRuntime?.inference), input.comparisonMode);
  validateDistinctResolvedVariants(owned, input.comparisonMode);
  for (const item of owned) await reference(item);
  await runWorkload({ variants: owned, contract, numericalGates: receipt.gates, phase,
    request: async (item, task, metadata) => {
    const file = path.join(item.id, `${task.id}-${metadata.repeat}-doppler.json`);
    const startedAt = new Date().toISOString();
    await phase('request-start', { variant: item.id, task: task.id, ...metadata, workloadPhase: metadata.phase });
    const attempt = { variant: item.id, kind: 'doppler', task: task.id, ...metadata, startedAt, file };
    receipt.attempts.push(attempt);
    await writeReceipt();
    let observation;
    try {
      observation = await bounded(item.page, async task => {
        const [{ getDevice }, { captureRequestTail }] = await Promise.all([
          import('/runtime/src/gpu/device.js'), import('/adapter/qwen-economics/request-tail.js'),
        ]);
        const device = getDevice();
        return captureRequestTail({ request: () => globalThis.REQUEST(task),
          drain: () => device.queue.onSubmittedWorkDone(), now: () => performance.now() });
      }, task, item.config.requestTimeoutMs);
    } catch (error) {
      const message = String(error.stack || error);
      observation = { rawResult: { error: message, controllerFailure: true },
        requestError: message, drainError: null, clockError: null, resourceTail: null };
    }
    attempt.finishedAt = new Date().toISOString();
    attempt.requestTail = observation.resourceTail;
    attempt.requestError = observation.requestError;
    attempt.drainError = observation.drainError;
    attempt.clockError = observation.clockError;
    const result = await retainRequestObservation(observation, async raw => {
      attempt.error = raw?.error ?? observation.requestError ?? observation.drainError ?? observation.clockError ?? null;
      await save(file, raw);
      await writeReceipt();
      await phase('request-end', { variant: item.id, task: task.id, ...metadata,
        workloadPhase: metadata.phase, error: attempt.error });
    });
    attempt.metrics = requestMetrics(result, metadata.phase !== 'quality');
    attempt.resourceMetrics = { ...observation.resourceTail,
      callerRequestOutputTokensPerSecond: result.tokenIds.length * 1000 / observation.resourceTail.callerRequestMs,
      callerRequestAndDrainOutputTokensPerSecond: result.tokenIds.length * 1000 / observation.resourceTail.callerRequestAndDrainMs };
    await writeReceipt();
    return result;
  } });
  receipt.acceptanceComplete = true;
  await writeReceipt();
  await phase('acceptance-complete');
  if (input.postAcceptanceCpuProfile === true) {
    receipt.diagnostics = [];
    for (const item of owned) {
      for (const task of contract.shared.throughputTasks) {
        await phase('post-acceptance-cpu-profile-start', { variant: item.id, task: task.id, eligibleForPerformanceAcceptance: false });
        const scope = await captureCpuProfile({ context: item.context, page: item.page, variant: item.id, taskId: task.id,
          request: () => bounded(item.page, task => globalThis.REQUEST(task), task, item.config.requestTimeoutMs),
          save: (name, value) => save(path.join(item.id, name), value) });
        receipt.diagnostics.push(scope);
        await writeReceipt();
        await phase('post-acceptance-cpu-profile-complete', { variant: item.id, task: task.id });
      }
    }
  }
  receipt.complete = true;
  await phase('run-complete');
} catch (error) {
  receipt.error = String(error.stack || error);
  process.exitCode = 1;
  await phase('rejected', { error: receipt.error });
} finally {
  await phase('cleanup-start', { timedOut });
  for (const item of [...owned].reverse()) {
    if (!timedOut) {
      try { await item.page?.evaluate(() => globalThis.CLOSE?.()); }
      catch (error) { receipt.cleanupErrors.push({ variant: item.id, stage: 'CLOSE', error: String(error) }); }
    }
    try { await item.context?.close(); }
    catch (error) { receipt.cleanupErrors.push({ variant: item.id, stage: 'context', error: String(error) }); }
    try { await closeProbeServer(item.server); }
    catch (error) { receipt.cleanupErrors.push({ variant: item.id, stage: 'server', error: String(error) }); }
    await save(path.join(item.id, 'logs.json'), item.logs);
  }
  receipt.finishedAt = new Date().toISOString();
  if (receipt.cleanupErrors.length) { receipt.complete = false; process.exitCode = 1; }
  await writeReceipt();
  await phase('cleanup-complete', { note: 'Owning wrapper must still retain GPU lock until all owned Chrome is gone; no automatic kill/reset.' });
}
