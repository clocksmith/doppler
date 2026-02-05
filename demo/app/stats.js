import { formatBytes } from '../../src/storage/quota.js';
import { state } from './state.js';
import { $ , setText } from './dom.js';
import { formatRate, formatMs, formatScalar } from './format.js';

export function getStatsMode() {
  if (state.uiMode === 'energy') return 'energy';
  if (state.uiMode === 'diffusion') return 'diffusion';
  if (state.uiMode === 'diagnostics' && state.lastDiagnosticsSuite === 'energy') {
    return 'energy';
  }
  if (state.uiMode === 'diagnostics' && state.lastDiagnosticsSuite === 'diffusion') {
    return 'diffusion';
  }
  const pipelineType = normalizeModelType(state.activePipeline?.manifest?.modelType);
  if (pipelineType === 'energy') return 'energy';
  if (pipelineType === 'diffusion') return 'diffusion';
  return 'text';
}

export function setStatLabels(labels) {
  setText($('stat-tps-label'), labels.tps);
  setText($('stat-ttft-label'), labels.ttft);
  setText($('stat-prefill-label'), labels.prefill);
  setText($('stat-e2e-label'), labels.e2e);
  setText($('stat-decode-label'), labels.decode);
  setText($('stat-tokens-label'), labels.tokens);
}

export function setRunLogLabels(labels) {
  setText($('run-log-ttft-label'), labels.ttft);
  setText($('run-log-prefill-label'), labels.prefill);
  setText($('run-log-decode-label'), labels.decode);
  setText($('run-log-e2e-label'), labels.e2e);
}

export function updatePerformancePanel(snapshot) {
  const tpsEl = $('stat-tps');
  const ttftEl = $('stat-ttft');
  const e2eEl = $('stat-e2e');
  const prefillEl = $('stat-prefill');
  const decodeEl = $('stat-decode');
  const tokensEl = $('stat-tokens');
  const mode = getStatsMode();

  const metrics = state.lastMetrics || {};
  const liveTps = state.runGenerating ? metrics.liveTokensPerSec : null;
  const tps = Number.isFinite(liveTps)
    ? liveTps
    : (Number.isFinite(metrics.tokensPerSec)
      ? metrics.tokensPerSec
      : (Number.isFinite(metrics.medianTokensPerSec) ? metrics.medianTokensPerSec : null));
  const stats = state.lastInferenceStats || {};
  if (mode === 'energy') {
    setStatLabels({
      tps: 'Steps/sec',
      ttft: 'Steps',
      prefill: 'Avg step',
      e2e: 'Energy',
      decode: 'Total',
      tokens: 'Shape',
    });

    const steps = Number.isFinite(stats.steps) ? stats.steps : null;
    const totalMs = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : null;
    const energy = Number.isFinite(stats.energy) ? stats.energy : null;
    const avgStepMs = (steps != null && totalMs && totalMs > 0) ? totalMs / steps : null;
    const stepsPerSec = (steps != null && totalMs && totalMs > 0)
      ? steps / (totalMs / 1000)
      : null;

    setText(tpsEl, stepsPerSec != null ? stepsPerSec.toFixed(2) : '--');
    setText(ttftEl, steps != null ? String(steps) : '--');
    setText(prefillEl, avgStepMs != null ? formatMs(avgStepMs) : '--');
    setText(decodeEl, formatMs(totalMs));
    setText(e2eEl, energy != null ? formatScalar(energy, 6) : '--');

    const request = state.lastEnergyRequest || {};
    const shape = Array.isArray(request.shape) ? request.shape : null;
    if (shape && shape.length) {
      setText(tokensEl, shape.join(' x '));
    } else if (
      Number.isFinite(request.height) &&
      Number.isFinite(request.width) &&
      Number.isFinite(request.channels)
    ) {
      setText(tokensEl, `${request.height}x${request.width}x${request.channels}`);
    } else {
      setText(tokensEl, '--');
    }
    return;
  }
  if (mode === 'diffusion') {
    setStatLabels({
      tps: 'Steps/sec',
      ttft: 'Prompt',
      prefill: 'Denoise',
      e2e: 'Total',
      decode: 'VAE',
      tokens: 'Resolution / Steps',
    });

    const steps = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
    const denoiseMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
    const promptMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
    const vaeMs = Number.isFinite(stats.vaeTimeMs) ? stats.vaeTimeMs : null;
    const totalMs = Number.isFinite(stats.totalTimeMs)
      ? stats.totalTimeMs
      : (Number.isFinite(promptMs) && Number.isFinite(denoiseMs)
        ? promptMs + denoiseMs + (Number.isFinite(vaeMs) ? vaeMs : 0)
        : null);
    const stepsPerSec = (steps != null && denoiseMs && denoiseMs > 0)
      ? steps / (denoiseMs / 1000)
      : null;

    setText(tpsEl, stepsPerSec != null ? stepsPerSec.toFixed(2) : '--');
    setText(ttftEl, formatMs(promptMs));
    setText(prefillEl, formatMs(denoiseMs));
    setText(decodeEl, formatMs(vaeMs));
    setText(e2eEl, formatMs(totalMs));

    const request = state.lastDiffusionRequest || {};
    const width = Number.isFinite(request.width) ? request.width : null;
    const height = Number.isFinite(request.height) ? request.height : null;
    const stepsLabel = steps != null ? steps : '--';
    if (width && height) {
      setText(tokensEl, `${width}x${height} / ${stepsLabel}`);
    } else if (steps != null) {
      setText(tokensEl, `-- / ${stepsLabel}`);
    } else {
      setText(tokensEl, '--');
    }
    return;
  }

  setStatLabels({
    tps: 'Tokens/sec',
    ttft: 'TTFT',
    prefill: 'Prefill',
    e2e: 'End-to-end',
    decode: 'Decode',
    tokens: 'Prompt / Gen',
  });

  setText(tpsEl, tps !== null ? `${tps.toFixed(2)}` : '--');

  const prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : null;
  const prefillTime = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
  const ttftMs = Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillTime;
  const prefillRate = (prefillTokens != null && prefillTime && prefillTime > 0)
    ? prefillTokens / (prefillTime / 1000)
    : null;
  const decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
  const decodeTime = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  const e2eTime = (Number.isFinite(stats.totalTimeMs) && stats.totalTimeMs > 0)
    ? stats.totalTimeMs
    : (Number.isFinite(prefillTime) && Number.isFinite(decodeTime) ? prefillTime + decodeTime : null);
  const e2eRate = (decodeTokens != null && e2eTime && e2eTime > 0)
    ? decodeTokens / (e2eTime / 1000)
    : null;
  if (ttftEl) {
    setText(ttftEl, formatMs(ttftMs));
  }
  if (e2eEl) {
    setText(e2eEl, formatRate(e2eRate));
  }
  if (prefillEl) {
    if (prefillTokens == null && ttftMs == null && prefillRate == null) {
      setText(prefillEl, '--');
    } else {
      const tokenLabel = prefillTokens != null ? `${prefillTokens} tok` : '--';
      const rateLabel = prefillRate != null ? `${prefillRate.toFixed(2)} tok/s` : '--';
      setText(prefillEl, `${tokenLabel} @ ${rateLabel}`);
    }
  }

  if (decodeEl) {
    if (decodeTokens == null && decodeTime == null) {
      setText(decodeEl, '--');
    } else {
      const tokenLabel = decodeTokens != null ? `${decodeTokens} tok` : '--';
      const rateLabel = (decodeTokens != null && decodeTime && decodeTime > 0)
        ? `${(decodeTokens / (decodeTime / 1000)).toFixed(2)} tok/s`
        : '--';
      setText(decodeEl, `${tokenLabel} - ${rateLabel}`);
    }
  }

  if (tokensEl) {
    if (prefillTokens == null && decodeTokens == null) {
      setText(tokensEl, '--');
    } else {
      const promptLabel = prefillTokens != null ? prefillTokens : '--';
      const genLabel = decodeTokens != null ? decodeTokens : '--';
      setText(tokensEl, `${promptLabel} / ${genLabel}`);
    }
  }
}

export function updateMemoryPanel(snapshot) {
  const poolStats = state.lastMemoryStats?.pool || null;
  const gpuStats = snapshot?.gpu || null;
  const gpuCurrent = Number.isFinite(gpuStats?.currentBytes) ? gpuStats.currentBytes : null;
  const gpuPeak = Number.isFinite(gpuStats?.peakBytes) ? gpuStats.peakBytes : null;
  const gpuRequested = Number.isFinite(gpuStats?.currentBytesRequested) ? gpuStats.currentBytesRequested : null;
  const activeBuffers = gpuStats?.activeBuffers ?? null;
  const pooledBuffers = gpuStats?.pooledBuffers ?? null;
  const gpuLimit = state.gpuMaxBytes || 0;

  setText($('stat-gpu-tracked'), Number.isFinite(gpuCurrent) ? formatBytes(gpuCurrent) : '--');
  setText($('stat-gpu-peak'), Number.isFinite(gpuPeak) ? formatBytes(gpuPeak) : '--');
  if (Number.isFinite(activeBuffers) && Number.isFinite(pooledBuffers)) {
    setText($('stat-gpu-buffers'), `${activeBuffers}/${pooledBuffers}`);
  } else {
    setText($('stat-gpu-buffers'), '--');
  }
  if (Number.isFinite(gpuRequested) && Number.isFinite(gpuCurrent)) {
    const requestedLabel = `${formatBytes(gpuRequested)} / ${formatBytes(gpuCurrent)}`;
    setText($('stat-gpu-requested'), requestedLabel);
  } else {
    setText($('stat-gpu-requested'), '--');
  }
  if (poolStats?.hitRate) {
    setText($('stat-gpu-hit'), poolStats.hitRate);
  } else {
    setText($('stat-gpu-hit'), '--');
  }
  if (gpuLimit) {
    setText($('stat-gpu-limit'), formatBytes(gpuLimit));
  } else {
    setText($('stat-gpu-limit'), '--');
  }

  const labelList = $('gpu-label-list');
  if (labelList) {
    const pool = state.activePipeline?.getBufferPool?.();
    const labelStats = typeof pool?.getLabelStats === 'function' ? pool.getLabelStats() : null;
    labelList.innerHTML = '';
    if (!labelStats || labelStats.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'type-caption';
      empty.textContent = 'No tracked buffers yet.';
      labelList.appendChild(empty);
    } else {
      const sorted = [...labelStats].sort((a, b) => (b.bytes || 0) - (a.bytes || 0));
      const top = sorted.slice(0, 6);
      for (const entry of top) {
        const row = document.createElement('div');
        row.className = 'stats-breakdown-row';

        const label = document.createElement('span');
        label.className = 'stats-breakdown-label';
        label.textContent = entry.label || 'unlabeled';

        const bytes = document.createElement('span');
        bytes.className = 'stats-breakdown-meta';
        bytes.textContent = Number.isFinite(entry.bytes) ? formatBytes(entry.bytes) : '--';

        const count = document.createElement('span');
        count.className = 'stats-breakdown-meta';
        count.textContent = Number.isFinite(entry.count) ? `${entry.count}` : '--';

        row.appendChild(label);
        row.appendChild(bytes);
        row.appendChild(count);
        labelList.appendChild(row);
      }
    }
  }

  const kvStats = state.lastMemoryStats?.kvCache || null;
  const kvAllocated = Number.isFinite(kvStats?.allocated) ? kvStats.allocated : null;
  const kvUsed = Number.isFinite(kvStats?.used) ? kvStats.used : null;
  const kvEff = Number.isFinite(kvStats?.efficiency) ? kvStats.efficiency : null;
  const kvSeq = Number.isFinite(kvStats?.seqLen) ? kvStats.seqLen : null;
  const kvMax = Number.isFinite(kvStats?.maxSeqLen) ? kvStats.maxSeqLen : null;
  const kvLayout = kvStats?.layout || null;

  setText($('stat-kv-allocated'), Number.isFinite(kvAllocated) ? formatBytes(kvAllocated) : '--');
  setText($('stat-kv-used'), Number.isFinite(kvUsed) ? formatBytes(kvUsed) : '--');
  if (Number.isFinite(kvEff)) {
    setText($('stat-kv-eff'), `${(kvEff * 100).toFixed(1)}%`);
  } else {
    setText($('stat-kv-eff'), '--');
  }
  if (Number.isFinite(kvSeq) && Number.isFinite(kvMax)) {
    setText($('stat-kv-seq'), `${kvSeq} / ${kvMax}`);
  } else {
    setText($('stat-kv-seq'), '--');
  }
  setText($('stat-kv-layout'), kvLayout || '--');

  const jsHeapUsed = Number.isFinite(snapshot?.jsHeapUsed) ? snapshot.jsHeapUsed : null;
  const jsHeapLimit = Number.isFinite(snapshot?.jsHeapLimit) ? snapshot.jsHeapLimit : null;
  if (Number.isFinite(jsHeapUsed) && Number.isFinite(jsHeapLimit) && jsHeapLimit > 0) {
    setText($('stat-heap'), `${formatBytes(jsHeapUsed)} / ${formatBytes(jsHeapLimit)}`);
  } else if (Number.isFinite(jsHeapUsed)) {
    setText($('stat-heap'), formatBytes(jsHeapUsed));
  } else {
    setText($('stat-heap'), '--');
  }

  if (state.systemMemoryBytes) {
    setText($('stat-ram-est'), formatBytes(state.systemMemoryBytes));
  } else {
    setText($('stat-ram-est'), '--');
  }

  const storageUsage = state.storageUsageBytes || 0;
  const storageQuota = state.storageQuotaBytes || 0;
  if (storageQuota) {
    setText($('stat-opfs'), `${formatBytes(storageUsage)} / ${formatBytes(storageQuota)}`);
  } else {
    setText($('stat-opfs'), '--');
  }
  setText($('stat-active-model'), state.activeModelId || 'none');
}

export function updateMemoryControls() {
  const unloadBtn = $('unload-model-btn');
  if (unloadBtn) {
    unloadBtn.disabled = !state.activePipeline;
  }
}

export function renderRunLog() {
  const container = $('run-log-rows');
  if (!container) return;
  const mode = getStatsMode();
  container.innerHTML = '';
  const entries = state.runLog.filter((entry) => entry.mode === mode);
  if (mode === 'diffusion') {
    setRunLogLabels({
      ttft: 'Prompt',
      prefill: 'Denoise',
      decode: 'VAE',
      e2e: 'Total',
    });
  } else if (mode === 'energy') {
    setRunLogLabels({
      ttft: 'Steps',
      prefill: 'Avg step',
      decode: 'Total',
      e2e: 'Energy',
    });
  } else {
    setRunLogLabels({
      ttft: 'TTFT',
      prefill: 'Prefill',
      decode: 'Decode',
      e2e: 'E2E',
    });
  }
  for (const entry of entries) {
    const row = document.createElement('div');
    row.className = 'run-log-row';
    let cells;
    if (mode === 'diffusion') {
      cells = [
        entry.label,
        formatMs(entry.promptMs),
        formatMs(entry.denoiseMs),
        formatMs(entry.vaeMs),
        formatMs(entry.totalMs),
      ];
    } else if (mode === 'energy') {
      cells = [
        entry.label,
        entry.steps != null ? String(entry.steps) : '--',
        formatMs(entry.avgStepMs),
        formatMs(entry.totalMs),
        entry.energy != null ? formatScalar(entry.energy, 6) : '--',
      ];
    } else {
      cells = [
        entry.label,
        formatMs(entry.ttftMs),
        entry.prefillRate != null ? formatRate(entry.prefillRate) : '--',
        entry.decodeRate != null ? formatRate(entry.decodeRate) : '--',
        entry.e2eRate != null ? formatRate(entry.e2eRate) : '--',
      ];
    }
    cells.forEach((text) => {
      const cell = document.createElement('span');
      cell.textContent = text;
      row.appendChild(cell);
    });
    container.appendChild(row);
  }
}

export function recordRunLog(stats, label, modeOverride) {
  if (!stats) return;
  const mode = modeOverride || getStatsMode();
  const entry = {
    mode,
    label,
  };
  let prefillTokens;
  let prefillTime;
  let decodeTokens;
  let decodeTime;
  let totalTime;
  if (mode === 'diffusion') {
    entry.promptMs = stats.prefillTimeMs;
    entry.denoiseMs = stats.decodeTimeMs;
    entry.vaeMs = stats.vaeTimeMs;
    entry.totalMs = stats.totalTimeMs;
  } else if (mode === 'energy') {
    entry.steps = stats.steps;
    entry.totalMs = stats.totalTimeMs;
    entry.avgStepMs = stats.totalTimeMs && stats.steps ? stats.totalTimeMs / stats.steps : null;
    entry.energy = stats.energy;
  } else {
    prefillTokens = Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : null;
    prefillTime = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
    decodeTokens = Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : null;
    decodeTime = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
    totalTime = Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : null;
    entry.ttftMs = Number.isFinite(stats.ttftMs) ? stats.ttftMs : prefillTime;
    entry.prefillRate = (prefillTokens != null && prefillTime && prefillTime > 0)
      ? prefillTokens / (prefillTime / 1000)
      : null;
    entry.decodeRate = (decodeTokens != null && decodeTime && decodeTime > 0)
      ? decodeTokens / (decodeTime / 1000)
      : null;
    entry.e2eRate = (decodeTokens != null && totalTime && totalTime > 0)
      ? decodeTokens / (totalTime / 1000)
      : null;
  }
  state.runLog.unshift(entry);
  state.runLog = state.runLog.slice(0, 8);
  renderRunLog();
}

function normalizeModelType(value) {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}
