import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawn, execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { createProbeServer, probeOrigin, closeProbeServer } from '../pooled-equal-weights/probe-server.js';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const contractBytes = await fs.readFile(config.contractPath);
const contract = JSON.parse(contractBytes);
const here = path.dirname(fileURLToPath(import.meta.url));
await fs.mkdir(config.outputDir, { recursive: false });
const save = (name, value) => fs.writeFile(path.join(config.outputDir, name), JSON.stringify(value, null, 2) + '\n');
const adapterSnapshot = path.join(config.outputDir, 'executed-source');
for (const name of ['qwen-economics/browser.js', 'pooled-equal-weights/shared-input.js',
  'pooled-equal-weights/probe-server.js', 'qwen-economics/run.js', 'qwen-economics/sample-resources.py',
  'qwen-economics/compare-reference-moe.py', 'qwen-economics/normalize-reference-manifest.js', 'qwen-economics/summarize.js', 'qwen-economics/economics.js']) {
  const target = path.join(adapterSnapshot, name);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.copyFile(path.join(path.dirname(here), name), target, 1);
}
await save('contract.json', contract);
await save('execution.json', { ...config, contractSha256: createHash('sha256').update(contractBytes).digest('hex'),
  runnerSha256: createHash('sha256').update(await fs.readFile(fileURLToPath(import.meta.url))).digest('hex'),
  browserAdapterSha256: createHash('sha256').update(await fs.readFile(path.join(here, 'browser.js'))).digest('hex') });
const evaluateBounded = (page, fn, arg) => { const limit = config.requestTimeoutMs; if (!Number.isFinite(limit) || limit <= 0) throw new Error("requestTimeoutMs required"); let timer; return Promise.race([page.evaluate(fn, arg), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Request exceeded requestTimeoutMs " + limit)), limit); })]).finally(() => clearTimeout(timer)); };
  let tasks = config.mode === 'quality' ? contract.shared.qualityTasks : config.mode === 'all'
    ? [...contract.shared.qualityTasks, ...contract.shared.throughputTasks] : contract.shared.throughputTasks;
  if (config.taskIds) tasks = tasks.filter(task => config.taskIds.includes(task.id));
  if (!tasks.length || config.taskIds && tasks.length !== new Set(config.taskIds).size) throw new Error('Requested task selection is empty or incomplete');
const owned = [], receipt = { startedAt: new Date().toISOString(), scope: contract.scope, attempts: [] };
const sampler = config.sampleResources ? spawn('python3', [path.join(adapterSnapshot, 'qwen-economics/sample-resources.py'),
  '--controller', String(process.pid), '--output', path.join(config.outputDir, 'resources.jsonl')],
{ stdio: ['ignore', 'ignore', 'inherit'] }) : null;
try {
  for (const kind of config.kinds) {
    const server = await createProbeServer({ entry: 'qwen-economics/browser.js',
      config: { ...config, contract, kind }, routes: [['/adapter/', adapterSnapshot],
        ['/runtime/', config.runtimeRoots?.[kind] ?? config.dopplerRoot], ['/pooled/', config.pooledRoot], ['/rdrr/', config.rdrrDir]],
      files: { '/model.gguf': config.ggufPath } });
    const item = { kind, server, context: null, page: null, logs: [] }; owned.push(item);
    const context = await chromium.launchPersistentContext(path.join(config.profileRoot, path.basename(config.outputDir), kind), {
      headless: true, executablePath: config.browserExecutable, args: config.browserArgs });
    const page = await context.newPage(), logs = [];
    Object.assign(item, { context, page, logs });
    page.on('console', value => {
      const line = `${value.type()}: ${value.text()}`; logs.push(line);
      fs.appendFile(path.join(config.outputDir, kind + '-live.log'), line + '\n').catch(() => {});
    });
    page.on('pageerror', error => {
      const line = 'PAGE ERROR: ' + String(error); logs.push(line);
      fs.appendFile(path.join(config.outputDir, kind + '-live.log'), line + '\n').catch(() => {});
    });
    await page.goto(probeOrigin(server));
    // Fail fast when the adapter page throws before READY instead of idling until loadTimeoutMs.
    let rejectPageError;
    const pageFailure = new Promise((_, reject) => { rejectPageError = reject; });
    page.on('pageerror', error => rejectPageError(new Error(kind + ' page error before READY: ' + String(error))));
    await Promise.race([page.waitForFunction(() => globalThis.READY !== undefined, null, { timeout: config.loadTimeoutMs }), pageFailure]);
    const ready = await page.evaluate(() => globalThis.READY);
    await save(kind + '-ready.json', ready); await save(kind + '-logs.json', logs);
    if (ready.error) throw new Error(kind + ': ' + ready.error);
    console.log('READY', kind, ready.loadMs);
  }
  if (config.referenceProbe) {
    const probe = config.referenceProbe;
    const item = owned.find(value => value.kind === 'doppler');
    if (!item || probe.contract.shared.maxNewTokens !== 2 || probe.observation.referenceDecodeSteps !== 1) {
      throw new Error('Exact two-output-token Doppler reference protocol required');
    }
    for (const field of ['contextTokens', 'sampling', 'promptFormat', 'eosTokenIds', 'requiredVendor']) {
      if (JSON.stringify(probe.contract.shared[field]) !== JSON.stringify(contract.shared[field])) {
        throw new Error('Reference phase changes shared field: ' + field);
      }
    }
    const probeDir = path.join(config.outputDir, 'reference-probe');
    await fs.mkdir(probeDir, { recursive: false });
    const saveProbe = (name, value) => fs.writeFile(path.join(probeDir, name), JSON.stringify(value, null, 2) + '\n');
    await saveProbe('contract.json', probe.contract);
    await saveProbe('execution.json', { ...config, phase: 'reference-probe',
      contractSha256: createHash('sha256').update(JSON.stringify(probe.contract)).digest('hex') });
    await saveProbe('doppler-ready.json', await item.page.evaluate(() => globalThis.READY));
    console.log('REFERENCE_START doppler');
    const task = probe.contract.shared.qualityTasks.find(value => value.id === 'arithmetic');
    if (!task) throw new Error('Reference arithmetic task missing');
    const result = await evaluateBounded(item.page, task => globalThis.REQUEST(task, 'reference-probe'), task);
    await saveProbe('arithmetic-0-doppler.json', result);
    await saveProbe('receipt.json', { complete: !result.error, scope: 'Completed reference phase only; main workload remains pending',
      attempts: [{ kind: 'doppler', task: 'arithmetic', repeat: 0, file: 'arithmetic-0-doppler.json', error: result.error ?? null }] });
    if (result.error) throw new Error(result.error);
    const gatePath = path.join(probeDir, 'numerical-gate.json');
    execFileSync('python3', [path.join(adapterSnapshot, 'qwen-economics/compare-reference-moe.py'),
      '--decode-steps', '1', '--reference', probe.referenceDir, '--gpu', probeDir,
      '--contract', probe.referenceContractPath, '--model', config.rdrrDir,
      '--runtime', config.runtimeRoots?.doppler ?? config.dopplerRoot, '--output', gatePath], { stdio: 'pipe' });
    const gate = JSON.parse(await fs.readFile(gatePath, 'utf8'));
    if (gate.passed !== true) throw new Error('Reference numerical gate rejected');
    receipt.referenceProbe = { complete: true, passed: true, gate: 'reference-probe/numerical-gate.json' };
    console.log('REFERENCE_PASS doppler', gate.checks.fullVocabulary.maxAbs);
  }
  if (config.layerProbe) {
    // Diagnostic only: per-engine single-token layer probe, before any task.
    for (const item of owned) {
      const result = await evaluateBounded(item.page, async () => globalThis.LAYER_PROBE(), null);
      await save(`layer-probe-${item.kind}.json`, result);
      console.log('LAYER_PROBE', item.kind, result.layers?.length ?? 0, result.error ?? '');
    }
  }
  for (const probe of config.prefixProbes ?? []) {
    // Diagnostic only: forced-prefix layer probes per engine, before any task.
    for (const item of owned) {
      const result = await evaluateBounded(item.page, async (value) => globalThis.PREFIX_PROBE(value), probe);
      await save(`prefix-probe-${probe.name}-${item.kind}.json`, result);
      console.log('PREFIX_PROBE', probe.name, item.kind, result.layers?.length ?? 0, result.error ?? '');
    }
  }
  for (const probe of config.routeProbes ?? []) {
    // Diagnostic only: per-engine router logits for every prompt position, before any task.
    for (const item of owned) {
      const result = await evaluateBounded(item.page, async (value) => globalThis.ROUTE_PROBE(value), probe);
      await save(`route-probe-${probe.name}-${item.kind}.json`, result);
      console.log('ROUTE_PROBE', probe.name, item.kind, result.promptLength ?? 0, result.duplicates ?? '', result.error ?? '');
    }
  }
  for (const task of tasks) {
    const qualityTask = task.expected !== undefined;
    const repetitions = config.repetitionsOverride ?? (qualityTask || config.captureLogitPositions ? 1 : contract.shared.warmupRuns + contract.shared.pairsPerPrompt);
    for (let repeat = 0; repeat < repetitions; repeat++) {
      const order = repeat % 2 ? [...owned].reverse() : owned;
      for (const item of order) {
        console.log('START', item.kind, task.id, repeat);
        const cpuProf = config.cpuProfileRepeats?.includes(repeat) && (config.cpuProfileKinds ?? ['doppler']).includes(item.kind);
        let cdp = null;
        if (cpuProf) { cdp = await item.context.newCDPSession(item.page); await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 200 }); await cdp.send('Profiler.start'); }
        let tracer = null, traceEvents = [];
        if (config.traceRepeats?.includes(repeat) && (config.cpuProfileKinds ?? ['doppler']).includes(item.kind)) {
          tracer = await item.context.newCDPSession(item.page);
          tracer.on('Tracing.dataCollected', ({ value }) => { for (const e of value) traceEvents.push(e); });
          await tracer.send('Tracing.start', { transferMode: 'ReportEvents', traceConfig: { includedCategories: ['v8', 'devtools.timeline', 'disabled-by-default-devtools.timeline', 'gpu', 'disabled-by-default-v8.compile', 'blink', 'toplevel', 'mojom', 'ipc'] } });
        }
        const result = await evaluateBounded(item.page, async task => globalThis.REQUEST(task), config.profileRepeats ? { ...task, profile: config.profileRepeats.includes(repeat) } : task);
        if (tracer) { const done = new Promise(r => tracer.once('Tracing.tracingComplete', r)); await tracer.send('Tracing.end'); await done;
          { const fh = await fs.open(path.join(config.outputDir, `${task.id}-${repeat}-${item.kind}.trace.jsonl`), 'w'); for (let i = 0; i < traceEvents.length; i += 5000) await fh.write(traceEvents.slice(i, i + 5000).map(e => JSON.stringify(e)).join('\n') + '\n'); await fh.close(); } await tracer.detach(); }
        if (cdp) { const { profile } = await cdp.send('Profiler.stop'); await fs.writeFile(path.join(config.outputDir, `${task.id}-${repeat}-${item.kind}.cpuprofile`), JSON.stringify(profile)); await cdp.detach(); }
        const file = `${task.id}-${repeat}-${item.kind}.json`;
        await save(file, result);
        receipt.attempts.push({ kind: item.kind, task: task.id, repeat,
          warmup: !qualityTask && repeat < contract.shared.warmupRuns,
          file, error: result.error ?? null });
        await save('receipt.json', receipt);
        console.log('DONE', item.kind, task.id, result.completeMs, result.qualityPass, result.error ?? '');
        if (result.error) throw new Error(result.error);
      }
    }
  }
  receipt.finishedAt = new Date().toISOString(); receipt.complete = true;
} catch (error) {
  receipt.error = String(error.stack || error); receipt.complete = false; process.exitCode = 1;
} finally {
  for (const item of owned.reverse()) {
    try { if (!receipt.error?.includes('requestTimeoutMs')) await item.page?.evaluate(() => globalThis.CLOSE?.()); } catch (error) { receipt.cleanupError = String(error); }
    await item.context?.close(); await closeProbeServer(item.server);
    await save(item.kind + '-logs.json', item.logs);
  }
  await save('receipt.json', receipt);
  sampler?.kill('SIGTERM');
}
