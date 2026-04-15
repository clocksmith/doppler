import fs from 'node:fs/promises';
import path from 'node:path';

export function toSummary(result) {
  if (!result || typeof result !== 'object') {
    return 'ok';
  }

  if (result.manifest?.modelId) {
    const contractStatus = result.executionContractArtifact?.ok === true
      ? ' contract=pass'
      : result.executionContractArtifact
        ? ' contract=fail'
        : '';
    return `converted ${result.manifest.modelId} (${result.tensorCount} tensors, ${result.shardCount} shards)${contractStatus}`;
  }

  if (result.kind === 'lora' || result.kind === 'distill') {
    const workloadId = result.workloadId || 'unknown';
    const action = result.action || 'run';
    const runRoot = result.runRoot || 'n/a';
    return `${result.kind} ${action} workload=${workloadId} runRoot=${runRoot}`;
  }

  const suite = result.suite || result.report?.suite || 'suite';
  const modelId = result.modelId || result.report?.modelId || 'unknown';
  const passed = Number.isFinite(result.passed) ? result.passed : null;
  const failed = Number.isFinite(result.failed) ? result.failed : null;
  const duration = Number.isFinite(result.duration) ? `${result.duration.toFixed(1)}ms` : 'n/a';
  if (passed !== null && failed !== null) {
    return `${suite} model=${modelId} passed=${passed} failed=${failed} duration=${duration}`;
  }
  return `${suite} model=${modelId}`;
}

export function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : 'n/a';
}

export function formatMs(value) {
  return Number.isFinite(value) ? `${Number(value).toFixed(1)}ms` : 'n/a';
}

export function quoteOneLine(value) {
  const s = String(value ?? '').replace(/\s+/g, ' ').trim();
  if (!s) return '""';
  const clipped = s.length > 120 ? `${s.slice(0, 117)}...` : s;
  return JSON.stringify(clipped);
}

export function quoteOneLineOrStructured(value) {
  if (typeof value === 'string') return quoteOneLine(value);
  if (value == null) return null;
  try {
    return quoteOneLine(JSON.stringify(value));
  } catch {
    return quoteOneLine(String(value));
  }
}

export function compactTimestamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, '');
}

export async function saveBenchResult(result, saveDir) {
  await fs.mkdir(saveDir, { recursive: true });
  const modelId = String(result?.modelId || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
  const ts = compactTimestamp();
  const filename = `${modelId}_${ts}.json`;
  const filePath = path.join(saveDir, filename);
  const json = JSON.stringify(result, null, 2);
  await fs.writeFile(filePath, json, 'utf-8');
  await fs.writeFile(path.join(saveDir, 'latest.json'), json, 'utf-8');
  return filePath;
}

export async function loadBaseline(comparePath, saveDir) {
  const resolved = comparePath === 'last'
    ? path.join(saveDir, 'latest.json')
    : path.resolve(comparePath);
  try {
    const raw = await fs.readFile(resolved, 'utf-8');
    return JSON.parse(raw);
  } catch (error) {
    console.error(`[compare] failed to load baseline from ${resolved}: ${error.message}`);
    return null;
  }
}

export function normalizeBenchMetrics(result) {
  const m = result?.metrics;
  if (!m) return m;
  return {
    decodeTokensPerSec: m.decodeTokensPerSec,
    prefillTokensPerSec: m.prefillTokensPerSec,
    firstTokenMs: m.firstTokenMs,
    firstResponseMs: m.firstResponseMs,
    prefillMs: m.prefillMs,
    decodeMs: m.decodeMs,
    totalRunMs: m.totalRunMs,
    modelLoadMs: m.modelLoadMs,
    decodeMsPerTokenP50: m.decodeMsPerTokenP50,
    decodeMsPerTokenP95: m.decodeMsPerTokenP95,
    decodeMsPerTokenP99: m.decodeMsPerTokenP99,
  };
}

export function compareBenchResults(current, baseline) {
  const cm = normalizeBenchMetrics(current);
  const bm = normalizeBenchMetrics(baseline);
  if (!cm || !bm) {
    console.error('[compare] missing metrics in current or baseline result');
    return { regressions: [], improvements: [] };
  }

  const isCrossEngine = (current?.env?.library) !== (baseline?.env?.library);
  const regressions = [];
  const improvements = [];

  const metrics = [
    { label: 'decode tok/s', cur: cm.decodeTokensPerSec, base: bm.decodeTokensPerSec, higherBetter: true },
    { label: 'prefill tok/s', cur: cm.prefillTokensPerSec, base: bm.prefillTokensPerSec, higherBetter: true },
    { label: 'first token', cur: cm.firstTokenMs, base: bm.firstTokenMs, higherBetter: false },
    { label: 'prefill ms', cur: cm.prefillMs, base: bm.prefillMs, higherBetter: false },
    { label: 'decode ms', cur: cm.decodeMs, base: bm.decodeMs, higherBetter: false },
    { label: 'first response', cur: cm.firstResponseMs, base: bm.firstResponseMs, higherBetter: false },
    { label: 'total run', cur: cm.totalRunMs, base: bm.totalRunMs, higherBetter: false },
    { label: 'model load', cur: cm.modelLoadMs, base: bm.modelLoadMs, higherBetter: false },
  ];

  // GPU phase metrics only available in Doppler-vs-Doppler comparisons
  const cg = current?.metrics?.gpu;
  const bg = baseline?.metrics?.gpu;
  if (cg && bg) {
    metrics.push(
      { label: 'gpu record ms', cur: cg.decodeRecordMs?.median, base: bg.decodeRecordMs?.median, higherBetter: false },
      { label: 'gpu submit_wait', cur: cg.decodeSubmitWaitMs?.median, base: bg.decodeSubmitWaitMs?.median, higherBetter: false },
      { label: 'gpu readback', cur: cg.decodeReadbackWaitMs?.median, base: bg.decodeReadbackWaitMs?.median, higherBetter: false },
    );
  }

  const curLabel = isCrossEngine ? (current?.env?.library || 'current') : 'current';
  const baseLabel = isCrossEngine ? (baseline?.env?.library || 'baseline') : 'baseline';
  const baseModelId = baseline.modelId || 'unknown';
  console.log(`[compare] vs ${baseLabel} model=${baseModelId}`);
  console.log(`[compare] ${'metric'.padEnd(20)} ${baseLabel.padStart(14)} ${curLabel.padStart(14)} ${'delta'.padStart(10)}`);

  for (const m of metrics) {
    if (!Number.isFinite(m.cur) || !Number.isFinite(m.base) || m.base === 0) continue;
    const deltaPct = ((m.cur - m.base) / Math.abs(m.base)) * 100;
    const sign = deltaPct >= 0 ? '+' : '';
    const deltaStr = `${sign}${deltaPct.toFixed(1)}%`;
    const isRegression = m.higherBetter ? deltaPct < -10 : deltaPct > 10;
    const isImprovement = m.higherBetter ? deltaPct > 10 : deltaPct < -10;
    const flag = isRegression ? ' !!REGRESSION' : isImprovement ? ' *improved' : '';
    console.log(`[compare] ${m.label.padEnd(20)} ${formatNumber(m.base, 1).padStart(14)} ${formatNumber(m.cur, 1).padStart(14)} ${deltaStr.padStart(10)}${flag}`);
    if (isRegression) regressions.push(m.label);
    if (isImprovement) improvements.push(m.label);
  }

  if (regressions.length) {
    console.log(`[compare] ${regressions.length} regression(s) detected (>10% threshold)`);
  }
  return { regressions, improvements };
}

export function printManifestSummary(results) {
  const completed = results.filter((r) => r.response && !r.error);
  const failed = results.filter((r) => r.error);
  console.log(`[sweep] ${completed.length} completed, ${failed.length} failed`);

  for (const r of results) {
    if (r.error) {
      console.log(`  ${r.label.padEnd(30)} FAILED`);
      continue;
    }
    const m = r.response?.result?.metrics;
    if (!m) {
      console.log(`  ${r.label.padEnd(30)} no metrics`);
      continue;
    }
    console.log(
      `  ${r.label.padEnd(30)} ` +
      `${formatNumber(m.decodeTokensPerSec)} decode tok/s  ` +
      `prefill=${formatNumber(m.prefillTokensPerSec)}  ` +
      `first=${formatMs(m.firstTokenMs)}`
    );
  }
}

function formatMB(bytes) {
  return Number.isFinite(bytes) ? `${(bytes / (1024 * 1024)).toFixed(1)}MB` : 'n/a';
}

export function printDeviceInfo(result) {
  const info = result?.deviceInfo;
  if (!info) return;
  const ai = info.adapterInfo;
  if (ai) {
    console.log(`[device] vendor=${ai.vendor || 'unknown'} arch=${ai.architecture || 'unknown'} device=${ai.device || 'unknown'}`);
  }
  console.log(
    `[device] f16=${info.hasF16 ? 'yes' : 'no'} subgroups=${info.hasSubgroups ? 'yes' : 'no'} timestamp_query=${info.hasTimestampQuery ? 'yes' : 'no'}`
  );
}

export function printGpuPhases(metrics) {
  if (typeof metrics?.decodeMode === 'string' && metrics.decodeMode.length > 0) {
    const reason = typeof metrics?.batchGuardReason === 'string' && metrics.batchGuardReason.length > 0
      ? ` guard=${metrics.batchGuardReason}`
      : '';
    console.log(`[decode] mode=${metrics.decodeMode}${reason}`);
  }
  const gpu = metrics?.gpu;
  if (!gpu) return;
  const rm = gpu.decodeRecordMs?.median;
  const sw = gpu.decodeSubmitWaitMs?.median;
  const rw = gpu.decodeReadbackWaitMs?.median;
  if (Number.isFinite(rm) || Number.isFinite(sw) || Number.isFinite(rw)) {
    console.log(`[gpu] decode record=${formatMs(rm)} submit_wait=${formatMs(sw)} readback_wait=${formatMs(rw)} (median)`);
  }
  const pm = gpu.prefillMs?.median;
  const dm = gpu.decodeMs?.median;
  if (Number.isFinite(pm) || Number.isFinite(dm)) {
    console.log(`[gpu] prefill=${formatMs(pm)} decode=${formatMs(dm)} (median gpu time)`);
  }
  const sts = gpu.singleTokenSubmitWaitMs?.median;
  const str = gpu.singleTokenReadbackWaitMs?.median;
  const sto = gpu.singleTokenOrchestrationMs?.median;
  if (Number.isFinite(sts) || Number.isFinite(str) || Number.isFinite(sto)) {
    console.log(
      `[gpu] single-token submit=${formatMs(sts)} readback=${formatMs(str)} ` +
      `orchestration=${formatMs(sto)} (median)`
    );
  }
  const batching = metrics?.batching;
  const batchedCalls = batching?.batchedForwardCalls?.median;
  const unbatchedCalls = batching?.unbatchedForwardCalls?.median;
  const batchedTime = batching?.totalBatchedTimeMs?.median;
  const unbatchedTime = batching?.totalUnbatchedTimeMs?.median;
  const submissions = batching?.gpuSubmissions?.median;
  if (
    Number.isFinite(batchedCalls)
    || Number.isFinite(unbatchedCalls)
    || Number.isFinite(batchedTime)
    || Number.isFinite(unbatchedTime)
    || Number.isFinite(submissions)
  ) {
    console.log(
      `[batching] batched_calls=${formatNumber(batchedCalls)} unbatched_calls=${formatNumber(unbatchedCalls)} ` +
      `batched_time=${formatMs(batchedTime)} unbatched_time=${formatMs(unbatchedTime)} ` +
      `gpu_submissions=${formatNumber(submissions)}`
    );
  }
}

export function printMemoryReport(result) {
  const mem = result?.memoryStats;
  if (!mem) return;
  const parts = [`used=${formatMB(mem.used)}`];
  if (mem.pool && Number.isFinite(mem.pool.currentBytesAllocated)) {
    parts.push(`pool=${formatMB(mem.pool.currentBytesAllocated)}`);
  }
  if (mem.kvCache) {
    parts.push(`kv_cache=${formatMB(mem.kvCache.allocated)}`);
    if (Number.isFinite(mem.kvCache.seqLen) && Number.isFinite(mem.kvCache.maxSeqLen)) {
      parts.push(`(seq=${mem.kvCache.seqLen}/${mem.kvCache.maxSeqLen})`);
    }
  }
  console.log(`[memory] ${parts.join(' ')}`);
}

export function printExecutionContractSummary(result) {
  const artifact = result?.metrics?.executionContractArtifact;
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  const session = artifact.session && typeof artifact.session === 'object'
    ? artifact.session
    : null;
  const attentionPhases = artifact.steps?.attentionPhases && typeof artifact.steps.attentionPhases === 'object'
    ? artifact.steps.attentionPhases
    : null;
  const parts = [
    `status=${artifact.ok === true ? 'pass' : 'fail'}`,
    checks.length > 0 ? `checks=${passedChecks}/${checks.length}` : 'checks=n/a',
  ];
  if (session?.layout) {
    parts.push(`layout=${session.layout}`);
  }
  if (attentionPhases) {
    parts.push(
      `attn(prefill=${attentionPhases.prefill ?? 'n/a'},decode=${attentionPhases.decode ?? 'n/a'},both=${attentionPhases.both ?? 'n/a'})`
    );
  }
  console.log(`[contract] ${parts.join(' ')}`);
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 3)) {
      console.log(`[contract] error=${quoteOneLine(error)}`);
    }
  }
}

function printSimpleArtifactSummary(label, artifact) {
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  console.log(
    `[${label}] status=${artifact.ok === true ? 'pass' : 'fail'} ` +
    `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'}`
  );
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 2)) {
      console.log(`[${label}] error=${quoteOneLine(error)}`);
    }
  }
}

export function printConvertContractSummary(result) {
  const artifact = result?.executionContractArtifact;
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  const session = artifact.session && typeof artifact.session === 'object'
    ? artifact.session
    : null;
  console.log(
    `[contract] status=${artifact.ok === true ? 'pass' : 'fail'} ` +
    `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'} ` +
    `layout=${session?.layout ?? 'n/a'}`
  );
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 3)) {
      console.log(`[contract] error=${quoteOneLine(error)}`);
    }
  }
  printSimpleArtifactSummary('layer-pattern', result?.layerPatternContractArtifact);
  printSimpleArtifactSummary('required-inference', result?.requiredInferenceFieldsArtifact);
}

export function printConvertReportSummary(result) {
  const reportInfo = result?.reportInfo;
  if (!reportInfo || typeof reportInfo !== 'object') return;
  if (typeof reportInfo.path !== 'string' || reportInfo.path.length === 0) return;
  console.log(`[report] ${reportInfo.path}`);
}

export function printMetricsSummary(result) {
  if (!result || typeof result !== 'object') return;
  if (result.kind === 'distill') {
    const stageCount = Array.isArray(result.stageResults) ? result.stageResults.length : 0;
    console.log(
      `[metrics] kind=distill action=${result.action || 'run'} stages=${stageCount} runRoot=${quoteOneLine(result.runRoot)}`
    );
    return;
  }
  if (result.kind === 'lora') {
    const exportCount = Array.isArray(result.exports) ? result.exports.length : 0;
    console.log(
      `[metrics] kind=lora action=${result.action || 'run'} exports=${exportCount} runRoot=${quoteOneLine(result.runRoot)}`
    );
    return;
  }
  const suite = String(result.suite || '');
  const metrics = result.metrics;
  if (!metrics || typeof metrics !== 'object') return;

  if (suite === 'inference' || suite === 'debug') {
    const prompt = quoteOneLine(metrics.prompt);
    console.log(`[metrics] prompt=${prompt}`);
    console.log(
      `[metrics] load=${formatMs(metrics.modelLoadMs)} ` +
      `prefillTokens=${Number.isFinite(metrics.prefillTokens) ? Math.round(metrics.prefillTokens) : 'n/a'} ` +
      `decodeTokens=${Number.isFinite(metrics.decodeTokens) ? Math.round(metrics.decodeTokens) : 'n/a'} ` +
      `maxTokens=${Number.isFinite(metrics.maxTokens) ? Math.round(metrics.maxTokens) : 'n/a'}`
    );
    console.log(
      `[metrics] first=${formatMs(metrics.firstTokenMs)} prefill=${formatMs(metrics.prefillMs)} ` +
      `decode=${formatMs(metrics.decodeMs)} total=${formatMs(metrics.totalRunMs)}`
    );
    console.log(
      `[metrics] tok/s=${formatNumber(metrics.decodeTokensPerSec)} ` +
      `prefill=${formatNumber(metrics.prefillTokensPerSec)} ` +
      `decode=${formatNumber(metrics.decodeTokensPerSec)}`
    );
    if (typeof result.output === 'string' && result.output.length > 0) {
      console.log(`[output] ${quoteOneLine(result.output)}`);
    }
    printExecutionContractSummary(result);
    return;
  }

  if (suite === 'bench') {
    if (Number.isFinite(metrics.embeddingDim) || Number.isFinite(metrics.avgEmbeddingMs)) {
      console.log(`[metrics] prompt=${quoteOneLine(metrics.prompt)}`);
      console.log(
        `[metrics] load=${formatMs(metrics.modelLoadMs)} runs=${Number.isFinite(metrics.warmupRuns) ? metrics.warmupRuns : 'n/a'}+${Number.isFinite(metrics.timedRuns) ? metrics.timedRuns : 'n/a'}`
      );
      console.log(
        `[metrics] embedding dim=${Number.isFinite(metrics.embeddingDim) ? Math.round(metrics.embeddingDim) : 'n/a'} ` +
        `median=${formatMs(metrics.medianEmbeddingMs)} avg=${formatMs(metrics.avgEmbeddingMs)} ` +
        `eps=${formatNumber(metrics.avgEmbeddingsPerSec)}`
      );
      printExecutionContractSummary(result);
        return;
    }

    console.log(`[metrics] prompt=${quoteOneLine(metrics.prompt)}`);
    console.log(
      `[metrics] load=${formatMs(metrics.modelLoadMs)} runs=${Number.isFinite(metrics.warmupRuns) ? metrics.warmupRuns : 'n/a'}+${Number.isFinite(metrics.timedRuns) ? metrics.timedRuns : 'n/a'} ` +
      `maxTokens=${Number.isFinite(metrics.maxTokens) ? Math.round(metrics.maxTokens) : 'n/a'}`
    );
    console.log(
      `[metrics] tokens prefill(avg)=${Number.isFinite(metrics.avgPrefillTokens) ? Math.round(metrics.avgPrefillTokens) : 'n/a'} ` +
      `decode(avg)=${Number.isFinite(metrics.avgDecodeTokens) ? Math.round(metrics.avgDecodeTokens) : 'n/a'} ` +
      `generated(avg)=${Number.isFinite(metrics.avgTokensGenerated) ? Math.round(metrics.avgTokensGenerated) : 'n/a'}`
    );
    console.log(
      `[metrics] decode tok/s=${formatNumber(metrics.decodeTokensPerSec)} avg=${formatNumber(metrics.avgDecodeTokensPerSec)} ` +
      `prefill=${formatNumber(metrics.prefillTokensPerSec)} avg=${formatNumber(metrics.avgPrefillTokensPerSec)}`
    );
    console.log(
      `[metrics] latency first=${formatMs(metrics.firstTokenMs)} ` +
      `prefill=${formatMs(metrics.prefillMs)} decode=${formatMs(metrics.decodeMs)}`
    );
    printExecutionContractSummary(result);
    printDeviceInfo(result);
    printGpuPhases(metrics);
    printMemoryReport(result);
    const samplePrompt = quoteOneLineOrStructured(metrics.promptInput);
    if (samplePrompt !== null) {
      console.log(`[sample] prompt=${samplePrompt}`);
    }
    if (typeof metrics.generatedText === 'string' && metrics.generatedText.length > 0) {
      console.log(`[sample] text=${quoteOneLine(metrics.generatedText)}`);
    }
    return;
  }

  if (suite === 'training') {
    const selectedTests = Array.isArray(metrics.selectedTests)
      ? metrics.selectedTests.length
      : 'n/a';
    const availableTests = Array.isArray(metrics.availableTests)
      ? metrics.availableTests.length
      : 'n/a';
    const stage = typeof metrics.trainingStage === 'string' ? metrics.trainingStage : 'n/a';
    console.log(
      `[metrics] training tests=${Number.isFinite(metrics.testsRun) ? metrics.testsRun : 'n/a'} ` +
      `selected=${selectedTests} available=${availableTests} stage=${stage}`
    );
  }
}
