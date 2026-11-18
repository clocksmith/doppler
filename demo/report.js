import { readRecordingJson } from './ui/recording-import.js';
import { state } from './ui/state.js';
import { getActivationRecording, importActivationRecording, isActivationRecordingFile,
  openActivationRecording, clearActivationRecording } from './ui/activation-grid/index.js';
import { restoreConversationHistory } from './input.js';
import { updateXrayPanels } from './ui/xray/index.js';
import {
  setFinalStats,
  setPhase,
  showWordQuality,
} from './output.js';

function $(id) { return document.getElementById(id); }

export function buildReport() {
  if (state.lastImportedReport) return state.lastImportedReport;
  const run = state.lastRun;
  if (!run) return null;

  return {
    schema: 'doppler.demo-report/v1',
    timestamp: new Date().toISOString(),
    modelId: state.lastInspection?.fingerprint?.identity?.artifact?.modelId
      ?? (run.mode === 'sample-inspection' ? null : state.modelId),
    inspection: state.lastInspection ?? null,
    activationRecording: state.lastInspection?.activationTrace === getActivationRecording()?.trace
      ? getActivationRecording() : null,
    settings: { ...(run.config ?? state.settings) },
    preset: state.preset,
    generationMode: run.wordQuality?.enabled ? 'guided_quality' : 'plain',
    prefillMs: run.prefillMs ?? null,
    decodeMs: run.decodeMs ?? null,
    totalTokens: run.totalTokens ?? 0,
    tokPerSec: run.tokPerSec ?? null,
    tokens: run.tokens ?? [],
    output: run.output ?? '',
    prompt: run.prompt ?? null,
    promptInput: run.promptInput ?? null,
    conversation: run.conversation ?? null,
    perplexity: run.perplexity ?? null,
    wordQuality: run.wordQuality ?? null,
    observationPolicy: run.observationPolicy ?? null,
    comparisonFingerprint: run.comparisonFingerprint ?? null,
    config: run.config ?? null,
    kernelPath: run.kernelPath ?? null,
    memorySnapshot: run.memorySnapshot ?? null,
  };
}

export function getReportOutput(report) {
  if (typeof report?.output === 'string') return report.output;
  if (!Array.isArray(report?.tokens)) return '';
  return report.tokens.map((token) => {
    if (typeof token === 'string') return token;
    return typeof token?.text === 'string' ? token.text : '';
  }).join('');
}

function getReportPrompt(report) {
  if (typeof report?.prompt === 'string') return report.prompt;
  const messages = report?.conversation?.messages;
  if (!Array.isArray(messages)) return null;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user' && typeof messages[index].content === 'string') {
      return messages[index].content;
    }
  }
  return null;
}

export function validateImportedReport(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Report JSON must be an object.');
  }
  if (value.schema != null && value.schema !== 'doppler.demo-report/v1') {
    throw new Error(`Unsupported report schema: ${value.schema}`);
  }
  const output = getReportOutput(value);
  const hasRunStats = Number.isFinite(value.totalTokens)
    || Number.isFinite(value.prefillMs)
    || Number.isFinite(value.decodeMs);
  if (!output && !hasRunStats) {
    throw new Error('Report has no output or generation stats.');
  }
  return value;
}

export function importReportData(value) {
  if (state.generating || state.modelBusy || state.settingsBusy) throw new Error('Wait for the current run to finish before opening a recording.');
  const report = validateImportedReport(value);
  clearActivationRecording();
  if (report.activationRecording) importActivationRecording(report.activationRecording);
  const output = getReportOutput(report);
  state.lastImportedReport = report;
  state.lastInspection = report.inspection && typeof report.inspection === 'object'
    && !Array.isArray(report.inspection) ? report.inspection : null;
  state.lastInferenceStats = null;
  delete globalThis.__DOPPLER_DEMO_EVIDENCE__;
  $('xray-container')?.replaceChildren();
  state.lastRun = {
    mode: ['guided_quality', 'token_press'].includes(report.generationMode)
      ? 'guided-quality'
      : 'plain',
    prefillMs: Number.isFinite(report.prefillMs) ? report.prefillMs : null,
    decodeMs: Number.isFinite(report.decodeMs) ? report.decodeMs : null,
    totalTokens: Number.isFinite(report.totalTokens) ? report.totalTokens : 0,
    tokPerSec: Number.isFinite(report.tokPerSec) ? report.tokPerSec : null,
    tokens: Array.isArray(report.tokens) ? report.tokens : [],
    output,
    imported: true,
  };
  showWordQuality(false);
  if (Array.isArray(report.conversation?.messages)) {
    restoreConversationHistory(report.conversation.messages);
  } else {
    const prompt = getReportPrompt(report);
    restoreConversationHistory([
      ...(prompt ? [{ role: 'user', content: prompt }] : []),
      ...(output ? [{ role: 'assistant', content: output }] : []),
    ]);
  }
  updateXrayPanels();
  setPhase('Opened run file', { quiet: true });
  setFinalStats(state.lastRun);
  setExportEnabled(true);
  return report;
}

// One Open control takes either a run file or a standalone layer grid recording.
async function openFile(file, signal) {
  if (!file) return;
  const { value, byteLength } = await readRecordingJson(file, { signal });
  signal.throwIfAborted();
  if (isActivationRecordingFile(value)) {
    openActivationRecording(value, byteLength);
    setPhase('Opened layer grid recording', { quiet: true });
    return;
  }
  importReportData(value);
}

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportReport() {
  const report = buildReport();
  if (!report) return;
  const name = `doppler-${report.modelId || 'run'}-${Date.now()}.json`;
  downloadJSON(report, name);
  markRunUnsaved(false);
}

function warnOnUnload(event) { event.preventDefault(); event.returnValue = ''; }

export function markRunUnsaved(unsaved) {
  const notice = $('run-unsaved');
  if (notice) notice.hidden = !unsaved;
  if (unsaved) globalThis.addEventListener?.('beforeunload', warnOnUnload);
  else globalThis.removeEventListener?.('beforeunload', warnOnUnload);
}

export function setExportEnabled(enabled) {
  markRunUnsaved(false);
  if ($('export-summary')) $('export-summary').disabled = !enabled;
  const btn = $('export-btn');
  if (btn) btn.disabled = !enabled;
}

export function initReport() {
  $('export-btn')?.addEventListener('click', exportReport);
  $('export-summary')?.addEventListener('click', () => {
    const report = buildReport();
    if (!report) return;
    // Numeric allowlist: imported reports can contain arbitrary nested text or token IDs.
    const summary = { schema: 'doppler.demo-report/v1' };
    for (const key of ['totalTokens', 'prefillMs', 'decodeMs', 'tokPerSec']) {
      if (Number.isFinite(report[key])) summary[key] = report[key];
    }
    downloadJSON(summary, 'doppler-run-summary.json');
  });
  let request = null;
  const importButton = $('import-btn');
  const importFile = $('import-file');
  importButton?.addEventListener('click', () => request ? request.abort() : importFile?.click());
  importFile?.addEventListener('change', async () => {
    request?.abort();
    const operation = new AbortController(); request = operation;
    importButton.textContent = 'Cancel opening file';
    try {
      await openFile(importFile.files?.[0], operation.signal);
    } catch (error) {
      setPhase(operation.signal.aborted ? 'Opening cancelled.' : `Could not open file: ${error.message}`);
    } finally {
      if (request === operation) {
        request = null;
        importButton.textContent = 'Open file';
        importFile.value = '';
      }
    }
  });
}
