// Node half of the decision benchmark: one child process per run, memory sampled
// over the child, results written next to the browser runs.
//   node node-run.js <config.json>
// config: { outDir, csvDirectory, repoRoot, q4kModelDirectory, f16ModelDirectory, hfRoot,
//           lanes: [...], repetitions, testLimit, qualityLanes: [...], laneTimeoutMs }
import fs from 'node:fs/promises';
import { readFileSync, readdirSync } from 'node:fs';
import { spawn } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadSplit, stratifiedSample } from '../dataset.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const outDir = path.join(path.resolve(config.outDir), 'node');
await fs.mkdir(outDir, { recursive: true });
const seed = 'issue-triage-20261021';
const readCsv = (split) => fs.readFile(path.join(config.csvDirectory, `issues_${split}.csv`), 'utf8');
const strip = ({ index, repo, label, title, body }) => ({ index, repo, label, title, body });
const train = stratifiedSample(await loadSplit('train', { readLocal: readCsv }), { split: 'train', perCell: 6, seed }).map(strip);
const test = stratifiedSample(await loadSplit('test', { readLocal: readCsv }), { split: 'test', perCell: 10, seed }).map(strip);
const scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'decision-bench-node-'));
const runConfigPath = path.join(scratch, 'run-config.json');
await fs.writeFile(runConfigPath, JSON.stringify({ test, train, firstIssue: train[0] }));

function sample(pid) {
  if (process.platform !== 'linux') return null;
  try {
    const rss = Number(/VmRSS:\s+(\d+)/.exec(readFileSync(`/proc/${pid}/status`, 'utf8'))?.[1] ?? 0) * 1024;
    let gpu = 0;
    for (const fd of readdirSync(`/proc/${pid}/fdinfo`)) {
      const info = readFileSync(`/proc/${pid}/fdinfo/${fd}`, 'utf8');
      if (!info.includes('drm-driver:\tamdgpu')) continue;
      gpu += Number(/drm-memory-gtt:\s+(\d+)/.exec(info)?.[1] ?? 0) * 1024;
      gpu += Number(/drm-memory-vram:\s+(\d+)/.exec(info)?.[1] ?? 0) * 1024;
    }
    return { rss, gpu };
  } catch { return null; }
}

async function runOne(lane, rep) {
  const quality = rep === 0 && (config.qualityLanes ?? []).includes(lane);
  const laneConfigPath = path.join(scratch, `${lane}-${rep}.json`);
  const output = path.join(scratch, `${lane}-${rep}.out.json`);
  await fs.writeFile(laneConfigPath, JSON.stringify({ ...config, benchmarkRoot: here, runConfigPath, quality }));
  const startedAt = Date.now();
  const child = spawn(process.execPath, [path.join(here, 'node-lane.js'), lane, laneConfigPath, output], { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', (chunk) => { stderr = (stderr + chunk).slice(-4000); });
  const peak = { totalRss: 0, gpuMemory: process.platform === 'linux' ? 0 : null };
  const timer = setInterval(() => {
    const s = sample(child.pid);
    if (!s) return;
    peak.totalRss = Math.max(peak.totalRss, s.rss);
    peak.gpuMemory = Math.max(peak.gpuMemory, s.gpu);
  }, 200);
  const kill = setTimeout(() => child.kill('SIGKILL'), config.laneTimeoutMs);
  const exit = await new Promise((resolve) => child.on('exit', (code, signal) => resolve({ code, signal })));
  clearInterval(timer);
  clearTimeout(kill);
  let report;
  try { report = JSON.parse(await fs.readFile(output, 'utf8')); } catch { report = { error: `no report (exit ${JSON.stringify(exit)})` }; }
  const record = { runId: `node-${lane}-${rep}`, browser: { name: 'node', kind: 'node', version: process.version }, lane, mode: 'node', rep, quality,
    launchToReportMs: Date.now() - startedAt, exit, stderrTail: stderr.slice(-1500), peakMemory: peak, report };
  await fs.writeFile(path.join(outDir, `${lane}-node-${rep}.json`), `${JSON.stringify(record, null, 1)}\n`);
  const t = report.test ?? [];
  console.log(`node ${lane} rep${rep}: ${report.error ? `ERROR ${report.error.split('\n')[0]}` : `cold ${report.coldCompleteMs?.toFixed(0)} ms, first ${report.firstDecisionMs?.toFixed(0)} ms, `
    + `test ${t.filter((d) => d.choice === d.label).length}/${t.length}, median ${[...t.map((d) => d.elapsedMs)].sort((a, b) => a - b)[t.length >> 1]?.toFixed(0)} ms, runtime ${JSON.stringify(report.runtime)}`}, RSS ${(peak.totalRss / 2 ** 30).toFixed(2)} GiB`);
}

try {
  for (let rep = 0; rep < config.repetitions; rep += 1) {
    const order = rep % 2 === 0 ? config.lanes : [...config.lanes].reverse();
    for (const lane of order) await runOne(lane, rep);
  }
} finally {
  await fs.rm(scratch, { recursive: true, force: true });
}
