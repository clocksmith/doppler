// In-page benchmark run for one lane. The page reports to the server that
// launched it, so any browser opened on this URL can be measured.
// URL parameters: lane, run, mode=cold|cpu, quality=1 (all test and train issues).
import { LANES } from './lanes.js';

const params = new URLSearchParams(location.search);
const lane = params.get('lane');
const runId = params.get('run');
const mode = params.get('mode') ?? 'cold';
const quality = params.get('quality') === '1';
const status = document.getElementById('status');
const messages = [];
for (const level of ['warn', 'error']) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    if (messages.length < 80) messages.push(`${level}: ${args.map((a) => String(a?.message ?? a)).join(' ').slice(0, 400)}`);
    original(...args);
  };
}
addEventListener('error', (event) => messages.length < 80 && messages.push(`pageerror: ${String(event.message).slice(0, 400)}`));
addEventListener('unhandledrejection', (event) => messages.length < 80 && messages.push(`rejection: ${String(event.reason?.message ?? event.reason).slice(0, 400)}`));

const report = { lane, runId, mode, quality, userAgent: navigator.userAgent, startedAtMs: performance.now() };
let lastProgress = 0;
const show = (text) => {
  status.textContent = text;
  // Heartbeat so the launcher can tell where a stalled run stopped.
  if (performance.now() - lastProgress > 5000 || !/^\S+: (test|train) /.test(text)) {
    lastProgress = performance.now();
    fetch(`/api/progress/${encodeURIComponent(runId)}`, { method: 'POST', body: `${Math.round(performance.now())} ${text}` }).catch(() => {});
  }
};

async function describeGpu() {
  if (!navigator.gpu) return { available: false };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { available: true, adapter: null };
  const info = adapter.info ?? {};
  return {
    available: true,
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, isFallbackAdapter: info.isFallbackAdapter ?? null },
    features: [...adapter.features].sort(),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

async function main() {
  report.gpu = await describeGpu();
  if (mode === 'cpu') {
    // Remove WebGPU before any engine loads; engines decide what to do without it.
    Object.defineProperty(Navigator.prototype, 'gpu', { get: () => undefined, configurable: true });
    report.gpuRemoved = navigator.gpu === undefined;
  }
  const config = await (await fetch('/api/run-config')).json();
  const definition = LANES[lane];
  if (!definition) throw new Error(`Unknown lane ${lane}`);
  report.comparison = definition.comparison;
  show(`${lane}: opening`);
  const openStart = performance.now();
  const opened = await definition.open({ descriptor: config.descriptor, hfBase: new URL('/hf/', location.href).href });
  report.openMs = performance.now() - openStart;
  report.runtime = opened.runtime;
  const decide = async (issue) => {
    const started = performance.now();
    const answer = await opened.decide(issue);
    return { index: issue.index, label: issue.label, repo: issue.repo, elapsedMs: performance.now() - started, ...answer };
  };
  const first = await decide(config.firstIssue);
  report.firstDecisionMs = first.elapsedMs;
  report.coldCompleteMs = performance.now();
  report.firstIssue = first;
  const testIssues = mode === 'cpu' ? config.test.slice(0, 30) : quality ? config.test : config.test.slice(0, 10);
  report.test = [];
  for (const [i, issue] of testIssues.entries()) {
    show(`${lane}: test ${i + 1}/${testIssues.length}`);
    report.test.push(await decide(issue));
  }
  if (quality && mode !== 'cpu') {
    report.train = [];
    for (const [i, issue] of config.train.entries()) {
      show(`${lane}: train ${i + 1}/${config.train.length}`);
      report.train.push(await decide(issue));
    }
  }
  await opened.close();
}

try {
  await main();
} catch (error) {
  report.error = `${error?.name ?? 'Error'}: ${error?.message ?? error}\n${error?.stack ?? ""}`.slice(0, 2000);
}
report.messages = messages;
report.finishedAtMs = performance.now();
show(report.error ? `${lane}: failed` : `${lane}: done`);
await fetch(`/api/result/${encodeURIComponent(runId)}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(report) });
