// Balanced alternating independent-process A/B runner and analysis for
// contract.json. Each sample is a separate Node process with its own Chrome.
import fs from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [baselineRoot, candidateRoot, modelDirectory, outDir, blocksArg, contractArg] = process.argv.slice(2);
if (!baselineRoot || !candidateRoot || !modelDirectory || !outDir) {
  throw new Error('usage: compare.js <baselineRoot> <candidateRoot> <modelDirectory> <outDir> [blocks=10] [contract.json]');
}
const blocks = Number(blocksArg ?? 10);
const probeScript = process.env.LM_HEAD_PROBE ?? 'probe.js';
const contractPath = path.resolve(contractArg ?? path.join(here, 'contract.json'));
const contract = JSON.parse(await fs.readFile(contractPath, 'utf8'));
const GPU_BUSY = '/sys/class/drm/card1/device/gpu_busy_percent';
const roots = { A: baselineRoot, B: candidateRoot };
await fs.mkdir(outDir, { recursive: true });

function readBusy() {
  if (process.platform !== 'linux') return null;
  try {
    return Number(readFileSync(GPU_BUSY, 'utf8').trim());
  } catch {
    return null;
  }
}

async function idleBusySample(ms) {
  const values = [];
  const end = Date.now() + ms;
  while (Date.now() < end) {
    values.push(readBusy());
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return values;
}

function runProbe(arm, output) {
  return new Promise((resolve) => {
    const busy = [];
    const timer = setInterval(() => busy.push(readBusy()), 100);
    const child = spawn(process.execPath, [path.join(here, probeScript), roots[arm], modelDirectory, contractPath, output], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stderr = '';
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    const kill = setTimeout(() => child.kill('SIGKILL'), 300000);
    child.on('exit', (code, signal) => {
      clearInterval(timer);
      clearTimeout(kill);
      resolve({ code, signal, busy, stderr: stderr.slice(-2000) });
    });
  });
}

const order = [];
for (let b = 0; b < blocks; b += 1) {
  order.push(...(b % 2 === 0 ? ['A', 'B', 'B', 'A'] : ['B', 'A', 'A', 'B']));
}

const samples = [];
for (let i = 0; i < order.length; i += 1) {
  const arm = order[i];
  const output = path.join(outDir, `sample-${String(i).padStart(3, '0')}-${arm}.json`);
  const preIdleBusy = await idleBusySample(1000);
  const run = await runProbe(arm, output);
  const probe = JSON.parse(await fs.readFile(output, 'utf8'));
  const sample = { index: i, block: Math.floor(i / 4), arm, output: path.basename(output), preIdleBusy, ...run, probe };
  samples.push(sample);
  const warm = probe.requests?.filter((r) => r.kind === 'warm').map((r) => r.elapsedMs) ?? [];
  console.log(`${i} ${arm} exit=${run.code} warmMedian=${median(warm)?.toFixed(1)} busyMax=${Math.max(...run.busy.filter((v) => v != null))} preIdleMax=${Math.max(...preIdleBusy.filter((v) => v != null))}`);
}

function median(values) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}
function sd(values) {
  const m = mean(values);
  return Math.sqrt(values.reduce((a, v) => a + (v - m) ** 2, 0) / (values.length - 1));
}
// Two-sided 97.5% t quantiles for small df.
const T975 = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.16, 14: 2.145, 15: 2.131, 16: 2.12, 17: 2.11, 18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045, 39: 2.023 };
function tq(df) {
  const keys = Object.keys(T975).map(Number).filter((k) => k <= df);
  return T975[Math.max(...keys)];
}
// Deterministic xorshift for reproducible bootstrap.
function rng(seed) {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
    return x / 4294967296;
  };
}
function bootstrapMean(values, iterations = 10000, seed = 20261021) {
  const next = rng(seed);
  const means = [];
  for (let i = 0; i < iterations; i += 1) {
    let sum = 0;
    for (let j = 0; j < values.length; j += 1) sum += values[Math.floor(next() * values.length)];
    means.push(sum / values.length);
  }
  means.sort((a, b) => a - b);
  return [means[Math.floor(0.025 * iterations)], means[Math.floor(0.975 * iterations)]];
}
const pct = (logRatio) => (Math.exp(logRatio) - 1) * 100;

function summarizeMetric(pick) {
  const pairs = [];
  for (let b = 0; b < blocks; b += 1) {
    const s = samples.filter((x) => x.block === b);
    // ABBA -> (s0,s1),(s3,s2); BAAB -> (s1,s0),(s2,s3): adjacent A/B pairs.
    const byPos = s.map((x) => ({ arm: x.arm, v: pick(x) }));
    for (const [i, j] of [[0, 1], [2, 3]]) {
      const a = byPos[i].arm === 'A' ? byPos[i].v : byPos[j].v;
      const c = byPos[i].arm === 'B' ? byPos[i].v : byPos[j].v;
      if (a != null && c != null) pairs.push(Math.log(c / a));
    }
  }
  const m = mean(pairs);
  const half = tq(pairs.length - 1) * sd(pairs) / Math.sqrt(pairs.length);
  const boot = bootstrapMean(pairs);
  return {
    pairs: pairs.length,
    meanPercent: pct(m),
    medianPercent: pct(median(pairs)),
    t95Percent: [pct(m - half), pct(m + half)],
    bootstrap95Percent: boot.map(pct),
    armMedians: {
      A: median(samples.filter((x) => x.arm === 'A').map(pick).filter((v) => v != null)),
      B: median(samples.filter((x) => x.arm === 'B').map(pick).filter((v) => v != null)),
    },
  };
}

const expected = samples.find((s) => s.probe.requests?.length)?.probe.requests[0];
const parity = samples.every((s) => s.probe.requests?.length === contract.design.requestsPerProcess.cold + contract.design.requestsPerProcess.warm
  && s.probe.requests.every((r) => r.text === expected.text && JSON.stringify(r.tokens) === JSON.stringify(expected.tokens)));
const matchesMacRetained = JSON.stringify(expected?.tokens) === JSON.stringify(contract.macRetainedExpectedTokens);
const lifecycle = samples.every((s) => s.code === 0 && !s.probe.error && s.probe.lifecycle?.unload && s.probe.lifecycle?.destroyDevice && s.probe.lifecycle?.browserClosed);
const warmMedian = (s) => median(s.probe.requests?.filter((r) => r.kind === 'warm').map((r) => r.elapsedMs) ?? []);
const warmDecode = (s) => median(s.probe.requests?.filter((r) => r.kind === 'warm').map((r) => r.stats.decodeTimeMs) ?? []);
const warmPrefill = (s) => median(s.probe.requests?.filter((r) => r.kind === 'warm').map((r) => r.stats.prefillTimeMs) ?? []);
const cold = (s) => s.probe.requests?.find((r) => r.kind === 'cold')?.elapsedMs ?? null;
const open = (s) => s.probe.openMs ?? null;
const metrics = {
  warmCompleteRequest: summarizeMetric(warmMedian),
  warmDecodeTimeMs: summarizeMetric(warmDecode),
  warmPrefillTimeMs: summarizeMetric(warmPrefill),
  coldCompleteRequest: summarizeMetric(cold),
  openMs: summarizeMetric(open),
};
const busyDuring = samples.map((s) => s.busy.filter((v) => v != null));
const contention = {
  preIdleNonzeroSamples: samples.filter((s) => s.preIdleBusy.some((v) => v > 0)).map((s) => s.index),
  maxPreIdleBusy: Math.max(...samples.flatMap((s) => s.preIdleBusy.filter((v) => v != null))),
  meanBusyDuringProbe: mean(busyDuring.map((v) => mean(v))),
};
const upper = metrics.warmCompleteRequest.bootstrap95Percent[1];
const summary = {
  contract: contract.schema,
  blocks,
  order,
  parity,
  matchesMacRetained,
  expectedText: expected?.text,
  lifecycle,
  metrics,
  contention,
  browserVersions: [...new Set(samples.map((s) => s.probe.browserVersion))],
  failedResponses: [...new Set(samples.flatMap((s) => s.probe.failedResponses ?? []))],
  promotion: {
    parity,
    lifecycle,
    latencyUpperBoundPercent: upper,
    passes: parity && lifecycle && upper < 0,
  },
};
await fs.writeFile(path.join(outDir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
await fs.writeFile(path.join(outDir, 'samples.json'), `${JSON.stringify(samples.map(({ probe, ...rest }) => ({ ...rest, probe: { ...probe, requests: probe.requests?.map(({ tokens, text, ...r }) => r) } })), null, 2)}\n`);
console.log(JSON.stringify({ parity, matchesMacRetained, lifecycle, metrics, contention, promotion: summary.promotion }, null, 2));
