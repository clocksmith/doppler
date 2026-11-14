// Runs the device benchmark protocol against a Doppler runtime and returns one
// evidence object. No DOM here: the page renders what this reports.

import { judgeRun, validateProtocol } from './protocol.js';
import { createRun, serializeRun } from './shared-run.js';

export const EVIDENCE_SCHEMA = 'doppler.observatory.device-benchmark/v1';
export const DEFAULT_RUNTIME_URL = '/doppler/src/index-browser.js';

const plain = (value) => (value == null ? null : JSON.parse(JSON.stringify(value)));
const rate = (tokens, ms) => (Number.isFinite(tokens) && Number.isFinite(ms) && tokens > 0 && ms > 0 ? tokens / (ms / 1000) : null);

export async function describeDevice() {
  const device = {
    userAgent: navigator.userAgent,
    platform: navigator.userAgentData?.platform ?? navigator.platform ?? null,
    deviceMemoryGb: navigator.deviceMemory ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    webgpu: 'unavailable',
    adapter: null,
    limits: null,
  };
  if (!navigator.gpu) return device;
  device.webgpu = 'no-adapter';
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter(); } catch { /* reported as no adapter */ }
  if (!adapter) return device;
  device.webgpu = 'available';
  const info = adapter.info ?? {};
  device.adapter = { vendor: info.vendor || null, architecture: info.architecture || null, device: info.device || null, description: info.description || null };
  device.limits = { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
  return device;
}

// The settings that decide how tokens are batched and read back. Two runs with
// the same values here took the same route through the runtime.
export function executionSettings(stats) {
  const plan = stats?.executionPlan ?? null;
  const active = plan?.[plan?.finalActivePlanId] ?? plan?.primary ?? null;
  if (!stats || !active) return null;
  return {
    decodeMode: stats.decodeMode ?? null,
    kernelPathId: stats.kernelPathId ?? null,
    planId: plan.finalActivePlanId ?? null,
    planChanges: plan.transitions?.length ?? 0,
    activationDtype: active.activationDtype ?? null,
    batchSize: active.batchSize ?? null,
    maxBatchDecodeTokens: active.maxBatchDecodeTokens ?? null,
    readbackInterval: active.readbackInterval ?? null,
    readbackMode: active.readbackMode ?? null,
    disableCommandBatching: active.disableCommandBatching ?? null,
  };
}

function summarizeStats(stats, wallMs, tokenCount) {
  const decodeTokens = stats?.decodeTokens ?? Math.max(0, tokenCount - 1);
  return {
    wallMs,
    execution: executionSettings(stats),
    stopReason: stats?.stopReason ?? null,
    promptTokens: stats?.prefillTokens ?? null,
    timeToFirstTokenMs: stats?.ttftMs ?? null,
    decodeTokens,
    decodeTimeMs: stats?.decodeTimeMs ?? null,
    decodeTokensPerSecond: rate(decodeTokens, stats?.decodeTimeMs),
    wallTokensPerSecond: rate(tokenCount, wallMs),
  };
}

async function benchmarkTierRun(model, workload) {
  const started = performance.now();
  const receipt = await model.inspect.generate(workload.prompt, { policyId: workload.observationPolicy, generation: { ...workload.generation } });
  const wallMs = performance.now() - started;
  const stats = plain(receipt.generationEvidence?.stats);
  return {
    run: { path: 'benchmark-tier', generatedTokenIds: [...receipt.generatedTokenIds], outputText: receipt.outputText, receiptWallMs: receipt.wallTimingMs, sampling: plain(receipt.generationEvidence?.generationConfig), ...summarizeStats(stats, wallMs, receipt.generatedTokenIds.length), stats },
    receipt,
  };
}

async function plainRun(model, workload) {
  const started = performance.now();
  const evidence = await model.generateWithEvidence(workload.prompt, { ...workload.generation });
  const wallMs = performance.now() - started;
  const stats = plain(evidence.stats);
  return { path: 'plain', generatedTokenIds: [...evidence.tokenIds], outputText: evidence.outputText, sampling: plain(evidence.generationConfig), ...summarizeStats(stats, wallMs, evidence.tokenIds.length), stats };
}

// The streaming API an application would call. It reports text, not token IDs.
async function streamingRun(model, workload) {
  const started = performance.now();
  let firstChunkMs = null;
  let outputText = '';
  for await (const piece of model.generate(workload.prompt, { ...workload.generation })) {
    if (firstChunkMs === null) firstChunkMs = performance.now() - started;
    outputText += typeof piece === 'string' ? piece : (piece?.text ?? '');
  }
  return { path: 'streaming', wallMs: performance.now() - started, firstChunkMs, outputText };
}

// `runtime` is the visit's runtime choice (see runtimes.js). Without one the hosted runtime is used.
export async function runDeviceBenchmark({ protocol, flags, runtime = null, onStage = () => {} }) {
  const runtimeUrl = runtime?.moduleUrl ?? DEFAULT_RUNTIME_URL;
  validateProtocol(protocol);
  const workload = protocol.workload;
  const evidence = {
    schema: EVIDENCE_SCHEMA,
    protocol: { id: protocol.id, version: protocol.version, workload: plain(workload), reference: plain(protocol.reference), eligibility: plain(protocol.eligibility) },
    startedAt: new Date().toISOString(),
    finishedAt: null,
    page: { origin: location.origin, path: location.pathname, runtimeUrl },
    runtime: null,
    flags: plain(flags),
    device: null,
    // Browsers slow down hidden tabs, so a run that was hidden is not a timing.
    tabHiddenDuringRun: document.visibilityState !== 'visible',
    load: null,
    policy: null,
    fingerprint: null,
    // The first run as a shared run: the same object the Explore page opens.
    run: null,
    cold: null,
    warm: [],
    plainPath: null,
    streaming: null,
    failure: null,
    verdict: null,
  };
  let stage = 'device';
  let model = null;
  const onVisibility = () => { if (document.visibilityState !== 'visible') evidence.tabHiddenDuringRun = true; };
  document.addEventListener('visibilitychange', onVisibility);
  const enter = (name, detail = null) => { stage = name; onStage({ stage: name, state: 'running', detail }); };
  const leave = (name, detail = null) => onStage({ stage: name, state: 'done', detail });
  try {
    if (!flags['benchmark.run']?.enabled) throw new Error('Running the benchmark is switched off in the flags file.');
    enter('device');
    evidence.device = await describeDevice();
    if (evidence.device.webgpu !== 'available') {
      throw new Error(evidence.device.webgpu === 'unavailable'
        ? 'This browser does not expose WebGPU, so no model can run here.'
        : 'This browser exposes WebGPU but returned no graphics adapter, so no model can run here.');
    }
    leave('device', [evidence.device.adapter.vendor, evidence.device.adapter.architecture, evidence.device.adapter.description].filter(Boolean).join(' · ') || null);

    enter('runtime');
    const module = await import(runtimeUrl);
    const dr = module.dr ?? module.doppler;
    let release = runtime?.release ?? null;
    if (!runtime) {
      try {
        const response = await fetch(new URL('../release.json', new URL(runtimeUrl, location.href)), { cache: 'no-store' });
        if (response.ok) { const body = await response.json(); release = { sourceCommit: body.runtime?.sourceCommit ?? body.sourceCommit ?? null, version: body.runtime?.version ?? null, label: body.runtime?.label ?? null, uncommittedSourceChanges: body.runtime?.uncommittedSourceChanges ?? null }; }
      } catch { /* a source checkout has no release file */ }
    }
    evidence.runtime = {
      version: module.DOPPLER_VERSION ?? null,
      // Which of this site's runtimes ran, and whether it is an unqualified preview build.
      channel: runtime?.id ?? 'hosted',
      preview: runtime?.preview === true,
      release,
    };
    leave('runtime', evidence.runtime.version);

    enter('load');
    const stored = async () => (await dr.listPersistentModels()).some((item) => item.modelId === workload.modelId);
    const load = { wasStored: null, storedAfter: null, wallMs: null, phases: [], fetchedModelFiles: false };
    evidence.load = load;
    try { load.wasStored = await stored(); } catch { /* unknown stays null */ }
    const loadStarted = performance.now();
    model = await dr.load(workload.modelId, {
      cache: 'opfs',
      runtimeProfile: workload.runtimeProfile,
      onProgress: (event) => {
        const phase = event?.phase ?? null;
        if (phase && load.phases.at(-1)?.phase !== phase) load.phases.push({ phase, atMs: performance.now() - loadStarted });
        if (phase && !['resolve', 'manifest'].includes(phase)) load.fetchedModelFiles = true;
        onStage({ stage: 'load', state: 'running', detail: event?.message ?? phase });
      },
    });
    load.wallMs = performance.now() - loadStarted;
    try { load.storedAfter = await stored(); } catch { /* unknown stays null */ }
    leave('load', `${(load.wallMs / 1000).toFixed(1)} s${load.wasStored ? ', from this browser\'s storage' : ''}`);

    enter('cold');
    const cold = await benchmarkTierRun(model, workload);
    evidence.cold = cold.run;
    const policy = cold.receipt.policy;
    evidence.policy = { id: policy.id, performanceRepresentative: policy.performanceRepresentative, modifiesExecution: policy.modifiesExecution, requiredCaptures: plain(policy.requiredCaptures) };
    evidence.fingerprint = plain(cold.receipt.fingerprint);
    evidence.run = JSON.parse(serializeRun(await createRun({
      receipt: cold.receipt,
      prompt: workload.prompt,
      environment: { runtimeVersion: evidence.runtime.version, runtimeCommit: evidence.runtime.release?.sourceCommit ?? null, runtimeChannel: evidence.runtime.channel, userAgent: evidence.device.userAgent },
    })));
    leave('cold', `${cold.run.generatedTokenIds.length} tokens`);

    // Warm and plain runs alternate so drift over time lands on both equally.
    const checkPlain = flags['benchmark.plainPathCheck']?.enabled === true;
    if (checkPlain) evidence.plainPath = [];
    for (let i = 0; i < workload.warmRuns; i++) {
      enter('warm', `run ${i + 1} of ${workload.warmRuns}`);
      evidence.warm.push((await benchmarkTierRun(model, workload)).run);
      if (checkPlain) {
        enter('plain', `run ${i + 1} of ${workload.warmRuns}`);
        evidence.plainPath.push(await plainRun(model, workload));
      }
    }
    leave('warm');
    if (checkPlain) {
      leave('plain');
      enter('streaming');
      evidence.streaming = await streamingRun(model, workload);
      leave('streaming');
    }
  } catch (error) {
    evidence.failure = { stage, name: error?.name ?? 'Error', message: String(error?.message ?? error) };
    onStage({ stage, state: 'failed', detail: evidence.failure.message });
  } finally {
    document.removeEventListener('visibilitychange', onVisibility);
    if (model) await model.unload?.().catch(() => {});
  }
  evidence.finishedAt = new Date().toISOString();
  evidence.verdict = judgeRun(protocol, evidence);
  return evidence;
}
