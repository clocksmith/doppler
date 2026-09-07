#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';
import { observeDocumentSearchGpu, readDocumentSearchGpuObservation } from './document-search-gpu-observation.js';

const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const fixtureBytes = await fs.readFile(config.fixturePath);
const fixture = JSON.parse(fixtureBytes);
const qualifications = {};
for (const [name, filename] of Object.entries(config.qualifications)) {
  const report = JSON.parse(await fs.readFile(filename, 'utf8'));
  assert(report.passed && report.physicalExecution, 'Passing installed physical qualification required.');
  assert.equal(report.fixtureDigest, hashBytesSha256(fixtureBytes));
  qualifications[name] = report;
}
const control = qualifications.unlimited;
const candidate = qualifications.bounded;
assert.deepEqual(candidate.build.models, control.build.models);
assert.equal(candidate.build.installedPackage.sha256, control.build.installedPackage.sha256);
assert.deepEqual(candidate.hardware, control.hardware);
assert(config.order.every(name => qualifications[name]));
assert(Number.isSafeInteger(config.sampleIntervalMs) && config.sampleIntervalMs > 0);
await fs.mkdir(config.outputDir);
const report = { schema: 'doppler.document-search-retention-comparison/v1', passed: false, config,
  fixtureDigest: hashBytesSha256(fixtureBytes), installedPackage: control.build.installedPackage,
  models: control.build.models, hardware: control.hardware, runs: [],
  scope: 'Interleaved offline process restarts on one physical device, using retained installed applications and the frozen search corpus.',
  memoryDefinition: 'Renderer RSS from Linux /proc, sampled during opening. RSS includes shared mappings and is not unique system memory. Artifact counters exclude temporary reads, returned slices, loader and GPU allocations.',
  externalAdoption: false };

async function rendererPids(profile) {
  const records = [];
  for (const name of await fs.readdir('/proc')) {
    if (!/^\d+$/.test(name)) continue;
    try {
      const [status, command] = await Promise.all([fs.readFile(`/proc/${name}/status`, 'utf8'), fs.readFile(`/proc/${name}/cmdline`, 'utf8')]);
      records.push({ pid: Number(name), parent: Number(status.match(/^PPid:\s+(\d+)/m)[1]),
        root: command.includes('--user-data-dir=' + profile), renderer: command.includes('--type=renderer') });
    } catch (error) { if (!['ENOENT', 'ESRCH', 'EACCES'].includes(error.code)) throw error; }
  }
  const owned = new Set(records.filter(row => row.root).map(row => row.pid));
  for (let previous = -1; previous !== owned.size;) {
    previous = owned.size;
    for (const row of records) if (owned.has(row.parent)) owned.add(row.pid);
  }
  const pids = records.filter(row => owned.has(row.pid) && row.renderer).map(row => row.pid);
  assert(pids.length, 'Physical browser renderer process must be identified.');
  return pids;
}
async function rss(pids) {
  let bytes = 0;
  for (const pid of pids) {
    const status = await fs.readFile(`/proc/${pid}/status`, 'utf8');
    bytes += Number(status.match(/^VmRSS:\s+(\d+)/m)[1]) * 1024;
  }
  return bytes;
}
let context;
try {
  for (const name of config.order) {
    const qualification = qualifications[name];
    const profile = path.join(qualification.config.outputDir, 'profile');
    context = await chromium.launchPersistentContext(profile, { headless: true,
      args: qualification.config.launchArgs, timeout: config.timeoutMs,
      env: { ...process.env, TMPDIR: qualification.config.temporaryDirectory } });
    await context.setOffline(true);
    const page = context.pages()[0] ?? await context.newPage();
    page.setDefaultTimeout(config.timeoutMs);
    await page.addInitScript(observeDocumentSearchGpu);
    const requests = [];
    page.on('requestfailed', request => requests.push({ url: request.url(), failure: request.failure() }));
    await page.goto(qualification.origin + '/index.html');
    await page.waitForFunction(() => globalThis.documentSearch?.ready);
    const pids = await rendererPids(profile);
    const run = { name, index: report.runs.length, rendererRssBefore: await rss(pids), samples: [], queries: [] };
    let sampling = false;
    let sampleError;
    const timer = setInterval(async () => {
      if (sampling) return;
      sampling = true;
      try { run.samples.push({ at: performance.now(), rendererRss: await rss(pids) }); }
      catch (error) { sampleError = error; }
      finally { sampling = false; }
    }, config.sampleIntervalMs);
    const started = performance.now();
    try { run.opened = await page.evaluate(() => globalThis.documentSearch.open()); }
    finally { clearInterval(timer); }
    run.openMs = performance.now() - started;
    run.upload = await page.evaluate(readDocumentSearchGpuObservation);
    if (sampleError) throw sampleError;
    run.rendererRssLoaded = await rss(pids);
    run.rendererRssPeak = Math.max(run.rendererRssLoaded, ...run.samples.map(sample => sample.rendererRss));
    run.observations = await page.evaluate(() => globalThis.documentSearch.observations);
    run.retainedBytes = run.observations.filter(event => event.type === 'capsule-load-complete')
      .reduce((total, event) => total + event.artifactMetrics.retainedBytes, 0);
    for (const query of fixture.queries) {
      const start = performance.now();
      const result = await page.evaluate(query => globalThis.documentSearch.search(query), query.text);
      const expected = control.phases.offline.find(row => row.query.id === query.id).result;
      assert.equal(result.results[0].document.id, query.expectedTopId);
      assert.deepEqual(result.results.map(row => row.document.id), expected.results.map(row => row.document.id));
      const maxScoreDifference = Math.max(...result.results.map((row, index) => Math.abs(row.rerankScore - expected.results[index].rerankScore)));
      assert(maxScoreDifference <= fixture.acceptance.offlineScoreMaxAbs);
      run.queries.push({ queryId: query.id, elapsedMs: performance.now() - start, maxScoreDifference,
        resultDigest: computeCanonicalSha256(result.results.map(row => ({ id: row.document.id, score: row.rerankScore }))) });
    }
    assert.equal(requests.length, 0, 'Offline opening and search require no unavailable requests.');
    await page.evaluate(() => globalThis.documentSearch.close());
    await context.close(); context = null;
    report.runs.push(run);
    await fs.writeFile(path.join(config.outputDir, `run-${run.index}-${name}.json`), JSON.stringify(run, null, 2), { flag: 'wx' });
    console.log(JSON.stringify({ run: run.index, name, openMs: run.openMs, rendererRssPeak: run.rendererRssPeak, retainedBytes: run.retainedBytes }));
  }
  const median = values => { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; };
  report.summary = Object.fromEntries(Object.keys(qualifications).map(name => {
    const runs = report.runs.filter(run => run.name === name);
    return [name, { runs: runs.length, openMsMedian: median(runs.map(run => run.openMs)),
      rendererRssPeakMedian: median(runs.map(run => run.rendererRssPeak)), retainedBytesMedian: median(runs.map(run => run.retainedBytes)),
      firstQueryMsMedian: median(runs.map(run => run.queries[0].elapsedMs)) }];
  }));
  report.lowerRendererMemory = report.summary.bounded.rendererRssPeakMedian < report.summary.unlimited.rendererRssPeakMedian;
  report.lowerRetainedBytes = report.summary.bounded.retainedBytesMedian < report.summary.unlimited.retainedBytesMedian;
  assert(report.lowerRendererMemory && report.lowerRetainedBytes);
  report.passed = true;
} catch (error) { report.error = { message: error.message, stack: error.stack }; }
finally {
  try { await context?.close(); } catch (error) { report.cleanupError = error.message; report.passed = false; }
  await fs.writeFile(path.join(config.outputDir, 'comparison.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, summary: report.summary, error: report.error?.message }));
if (!report.passed) process.exitCode = 1;
