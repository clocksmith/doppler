import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { createProbeServer, closeProbeServer, probeOrigin } from './probe-server.js';
import { freezeResidentInputs, verifyResidentInputs, hashFile, hashJson, fileInventory } from './resident-snapshot.js';
import { admitResidentProofs, assertKnownAdapter } from './resident-admission.js';
import { validateResidentPlan, runResidentPairs } from './resident-pairs.js';
import { summarizeResidentPairs, validateResidentStatisticsMethod } from './resident-statistics.js';

export async function withDeadline(operation, timeoutMs, label) {
  let timer;
  try {
    return await Promise.race([operation, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} exceeded frozen timeoutMs=${timeoutMs}`)), timeoutMs);
    })]);
  } finally { clearTimeout(timer); }
}

export async function persistResidentRecord(output, record) {
  const directory = path.join(output, 'attempts');
  await fs.mkdir(directory, { recursive: true });
  const name = Number.isInteger(record.sequence) ? `request-${String(record.sequence).padStart(3, '0')}`
    : `event-${record.type}`;
  const file = await fs.open(path.join(directory, `${name}.json`), 'wx');
  try { await file.writeFile(JSON.stringify({ schema: 'doppler.equal-weight-resident-event.v1', recordedAt: new Date().toISOString(), ...record }, null, 2) + '\n'); await file.sync(); }
  finally { await file.close(); }
}

async function prepareResidentInputs(bytes) {
  const config = JSON.parse(bytes), output = path.resolve(config.outputDir);
  if (!Array.isArray(config.browserArgs) || !config.browserArgs.every(value => typeof value === 'string')
      || !Number.isSafeInteger(config.timeoutMs) || config.timeoutMs <= 0
      || !/^[a-z0-9-]+$/.test(config.requiredVendor)) throw new Error('Declare browserArgs, positive timeoutMs and requiredVendor');
  const identity = await freezeResidentInputs(config, bytes, output);
  const runtime = path.join(output, 'runtime');
  const contract = JSON.parse(await fs.readFile(path.join(runtime, 'benchmarks/vendors/pooled-equal-weights/contract-f32kv.json')));
  const method = JSON.parse(await fs.readFile(path.join(runtime, 'tools/pooled-equal-weights/resident-method-f32kv.json')));
  const [promptPath, pointer] = contract.shared.promptSource.split('#');
  // promptSource's retained contract is an input, never a replacement workload.
  if (await hashFile(path.join(config.dopplerRoot, promptPath)) !== method.promptSourceSha256) throw new Error('Frozen four-prompt source changed');
  const promptSource = JSON.parse(await fs.readFile(path.join(config.dopplerRoot, promptPath)));
  const frozenPrompts = pointer.split('/').reduce((value, key) => value[key], promptSource);
  const references = JSON.parse(await fs.readFile(config.referencesPath));
  if (hashJson(references.prompts.map(({ id, text }) => ({ id, text }))) !== hashJson(frozenPrompts)) throw new Error('Reference prompts differ from frozen contract workload');
  const prompts = references.prompts;
  validateResidentPlan(contract, method, prompts);
  validateResidentStatisticsMethod(method);
  const actualContractHash = await hashFile(path.join(runtime, method.contract));
  if (actualContractHash !== method.contractSha256) throw new Error('Frozen timing contract changed');
  const frozen = { ...config, outputDir: output, contract, method, prompts };
  const admission = await admitResidentProofs(frozen, identity, path.join(output, 'proofs'));
  await fs.writeFile(path.join(output, 'plan.json'), JSON.stringify({ contract, method, prompts,
    contractSha256: actualContractHash, methodSha256: await hashFile(path.join(runtime, 'tools/pooled-equal-weights/resident-method-f32kv.json')),
    referencesSha256: await hashFile(config.referencesPath), promptSourceSha256: await hashFile(path.join(config.dopplerRoot, promptPath)), admission }, null, 2));
  await fs.writeFile(path.join(output, 'frozen-config.json'), JSON.stringify(frozen, null, 2));
  await fs.writeFile(path.join(output, 'prepared-identity.json'), JSON.stringify({
    configSha256: await hashFile(path.join(output, 'frozen-config.json')),
    planSha256: await hashFile(path.join(output, 'plan.json')),
    sourceIdentitySha256: await hashFile(path.join(output, 'source-identity.json')),
    proofs: await fileInventory(path.join(output, 'proofs')),
  }, null, 2));
  return output;
}

export async function prepareResident(configPath) {
  const bytes = await fs.readFile(configPath, 'utf8'), config = JSON.parse(bytes);
  try { await fs.stat(config.outputDir); throw new Error('Never reuse a resident output directory'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  try { return await prepareResidentInputs(bytes); }
  catch (error) {
    if (config.outputDir && error.code !== 'EEXIST') {
      try { await fs.writeFile(path.join(config.outputDir, 'preparation-failure.json'), JSON.stringify({
        completed: false, error: String(error.stack || error), gpuLaunched: false,
      }, null, 2), { flag: 'wx' }); } catch { /* Preserve existing receipt or original preparation failure. */ }
    }
    throw error;
  }
}

async function executeFrozen(output) {
  const config = JSON.parse(await fs.readFile(path.join(output, 'frozen-config.json')));
  const identity = JSON.parse(await fs.readFile(path.join(output, 'source-identity.json')));
  const plan = JSON.parse(await fs.readFile(path.join(output, 'plan.json')));
  const prepared = JSON.parse(await fs.readFile(path.join(output, 'prepared-identity.json')));
  async function verifyPrepared() {
    if (prepared.configSha256 !== await hashFile(path.join(output, 'frozen-config.json'))
        || prepared.planSha256 !== await hashFile(path.join(output, 'plan.json'))
        || prepared.sourceIdentitySha256 !== await hashFile(path.join(output, 'source-identity.json'))
        || hashJson(prepared.proofs) !== hashJson(await fileInventory(path.join(output, 'proofs')))) throw new Error('Frozen plan/proof/config drift');
  }
  await verifyPrepared();
  await verifyResidentInputs(config, identity, output);
  if (hashJson(config.contract) !== hashJson(plan.contract) || hashJson(config.method) !== hashJson(plan.method)
      || hashJson(config.prompts) !== hashJson(plan.prompts) || plan.admission.pass !== true) throw new Error('Frozen plan changed before launch');
  const { chromium } = await import('playwright');
  let browser, failure;
  const owned = [], logs = [], sessions = {};
  try {
    browser = await chromium.launch({ executablePath: config.browserExecutable, args: config.browserArgs, headless: true });
    for (const kind of ['pooled', 'doppler']) {
      let context, page, server;
      sessions[kind] = {
        async ready() {
          server = await createProbeServer({ entry: 'resident-browser.js', config: { ...config, kind,
            pooledNativeOptions: config.contract.pooledOverlay.nativeOptions },
          routes: [['/adapter/', path.join(output, 'runtime/tools/pooled-equal-weights')],
            ['/vendor/', path.join(output, 'vendor')], ['/runtime/', path.join(output, 'runtime')],
            ['/canonical/', config.canonicalDir], ['/rdrr/', config.rdrrDir]], files: { '/model.gguf': config.ggufPath } });
          context = await browser.newContext();
          page = await context.newPage();
          page.on('console', message => logs.push({ kind, type: message.type(), text: message.text() }));
          page.on('pageerror', error => logs.push({ kind, type: 'pageerror', text: String(error) }));
          await page.goto(probeOrigin(server));
          await page.waitForFunction(() => globalThis.READY !== undefined, null, { timeout: config.timeoutMs });
          const ready = await page.evaluate(() => globalThis.READY);
          await persistResidentRecord(output, { type: `ready-${kind}`, ...ready });
          if (ready.error) throw new Error(ready.error);
          const materialization = ready.identity?.materialization;
          if (materialization?.pass !== true || materialization.checks.length !== config.method.expectedTensorCount
              || materialization.checks.some(check => check.pass !== true || check.actual !== check.expected)) throw new Error('Resident GPU materialization verification failed');
          const numericalAdapter = plan.admission.adapter;
          assertKnownAdapter(ready.adapter, numericalAdapter);
          if (ready.userAgent !== plan.admission.browserUserAgent) throw new Error('Numerical evidence and resident browser identity differ');
          owned.push({ kind, adapter: ready.adapter });
          if (owned.length === 2) {
            assertKnownAdapter(owned[0].adapter, owned[1].adapter);
            const cdp = await browser.newBrowserCDPSession();
            let systemInfo;
            try { systemInfo = await cdp.send('SystemInfo.getInfo'); } finally { await cdp.detach(); }
            await persistResidentRecord(output, { type: 'physical-context', systemInfo });
            if (owned.some(value => !value.adapter.device) && systemInfo.gpu.devices.length !== 1) {
              throw new Error('Browser hides device IDs and reports multiple GPUs; common physical adapter is unresolved');
            }
          }
        },
        async prepare(prompt) { await withDeadline(page.evaluate(prompt => globalThis.RESIDENT.prepare(prompt), prompt), config.timeoutMs, `${kind} reset/drain`); },
        async request(prompt) {
          const response = await withDeadline(page.evaluate(prompt => globalThis.RESIDENT.request(prompt), prompt), config.timeoutMs, `${kind} request`);
          if (response.error) { const error = new Error(response.error); error.partialRequest = response.partialRequest; throw error; }
          return response.result;
        },
        async close() {
          const errors = [];
          if (page) try { await withDeadline(page.evaluate(() => globalThis.RESIDENT?.close()), config.timeoutMs, `${kind} cleanup`); } catch (error) { errors.push(String(error)); }
          if (context) try { await context.close(); } catch (error) { errors.push(String(error)); }
          if (server) try { await closeProbeServer(server); } catch (error) { errors.push(String(error)); }
          if (errors.length) throw new Error(errors.join('\n'));
        },
      };
    }
    const attempts = await runResidentPairs({ sessions, contract: config.contract, method: config.method,
      prompts: config.prompts, record: record => persistResidentRecord(output, record) });
    await verifyResidentInputs(config, identity, output);
    if (logs.some(log => log.type === 'pageerror')) throw new Error('Browser page error retained in log');
    await verifyPrepared();
    const statistics = summarizeResidentPairs(attempts, config.prompts, config.contract, config.method);
    await persistResidentRecord(output, { type: 'summary', statistics, correctnessAndIdentityPass: true,
      performancePass: statistics.performancePass, conclusion: 'Scoped resident comparison only; no all-platform qualification.' });
  } catch (error) {
    failure = error;
    await persistResidentRecord(output, { type: 'failure', error: String(error.stack || error) });
  } finally {
    try { if (browser) await withDeadline(browser.close(), config.timeoutMs, 'browser cleanup'); }
    catch (error) { failure ??= error; await persistResidentRecord(output, { type: 'browser-cleanup-failure', error: String(error) }); }
    await fs.writeFile(path.join(output, 'browser.json'), JSON.stringify({ version: browser?.version(), logs }, null, 2));
  }
  if (failure) throw failure;
}

async function main() {
  const [action, target] = process.argv.slice(2);
  if (action === 'prepare') { console.log(await prepareResident(target)); return; }
  if (action !== 'run' && action !== 'execute-frozen') throw new Error('Usage: run-resident-pairs.js prepare <config.json> | run <prepared-output>');
  const output = path.resolve(target);
  const entry = path.join(output, 'runtime/tools/pooled-equal-weights/run-resident-pairs.js');
  if (action === 'execute-frozen') {
    if (fileURLToPath(import.meta.url) !== entry) throw new Error('Execute the frozen committed runner, never shared source');
    await executeFrozen(output);
  } else {
    const child = spawn(process.execPath, [entry, 'execute-frozen', output], { stdio: 'inherit' });
    await new Promise((resolve, reject) => { child.on('error', reject); child.on('exit', code => code === 0 ? resolve() : reject(new Error(`Frozen runner exited ${code}`))); });
  }
}
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) await main();
