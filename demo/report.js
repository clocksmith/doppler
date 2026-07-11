import { state } from './ui/state.js';
import { restoreConversationHistory } from './input.js';
import {
  renderImportedChat,
  setFinalStats,
  setPhase,
  showTokenPress,
} from './output.js';

function $(id) { return document.getElementById(id); }

export function buildReport() {
  if (state.lastImportedReport) return state.lastImportedReport;
  const run = state.lastRun;
  if (!run) return null;

  return {
    schema: 'doppler.demo-report/v1',
    timestamp: new Date().toISOString(),
    modelId: state.modelId,
    settings: { ...state.settings },
    preset: state.preset,
    generationMode: run.tokenPress?.enabled ? 'token_press' : 'plain',
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
    tokenPress: run.tokenPress ?? null,
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
  const report = validateImportedReport(value);
  const output = getReportOutput(report);
  state.lastImportedReport = report;
  state.lastRun = {
    mode: report.generationMode === 'token_press' ? 'token-press' : 'plain',
    prefillMs: Number.isFinite(report.prefillMs) ? report.prefillMs : null,
    decodeMs: Number.isFinite(report.decodeMs) ? report.decodeMs : null,
    totalTokens: Number.isFinite(report.totalTokens) ? report.totalTokens : 0,
    tokPerSec: Number.isFinite(report.tokPerSec) ? report.tokPerSec : null,
    tokens: Array.isArray(report.tokens) ? report.tokens : [],
    output,
    imported: true,
  };
  showTokenPress(false);
  if (Array.isArray(report.conversation?.messages)) {
    restoreConversationHistory(report.conversation.messages, {
      historyEnabled: report.conversation.historyEnabled,
      turnLimit: report.conversation.turnLimit,
    });
  } else {
    renderImportedChat(output, getReportPrompt(report));
  }
  setPhase('Imported report');
  setFinalStats(state.lastRun);
  setExportEnabled(true);
  return report;
}

async function importReportFile(file) {
  if (!file) return null;
  const source = await file.text();
  return importReportData(JSON.parse(source));
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
}

export function exportReferenceTranscript() {
  const transcript = state.lastReferenceTranscript;
  if (!transcript) return;
  const modelId = state.modelId || 'run';
  const name = `doppler-reference-transcript-${modelId}-${Date.now()}.json`;
  downloadJSON(transcript, name);
}

export function setExportEnabled(enabled) {
  const btn = $('export-btn');
  if (btn) btn.disabled = !enabled;
}

export function setTranscriptExportEnabled(enabled) {
  const btn = $('export-transcript-btn');
  if (btn) btn.disabled = !enabled;
}

export function initReport() {
  $('export-btn')?.addEventListener('click', exportReport);
  $('export-transcript-btn')?.addEventListener('click', exportReferenceTranscript);
  const importButton = $('import-btn');
  const importFile = $('import-file');
  importButton?.addEventListener('click', () => importFile?.click());
  importFile?.addEventListener('change', async () => {
    try {
      await importReportFile(importFile.files?.[0]);
    } catch (error) {
      setPhase(`Import failed: ${error.message}`);
    } finally {
      importFile.value = '';
    }
  });
  const captureToggle = $('capture-transcript-toggle');
  if (captureToggle) {
    captureToggle.checked = state.captureTranscriptEnabled === true;
    captureToggle.addEventListener('change', (e) => {
      state.captureTranscriptEnabled = e.target.checked === true;
    });
  }
}
