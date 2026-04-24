import { state } from './ui/state.js';

function $(id) { return document.getElementById(id); }

function buildReport() {
  const run = state.lastRun;
  if (!run) return null;

  return {
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
    perplexity: run.perplexity ?? null,
    tokenPress: run.tokenPress ?? null,
    config: run.config ?? null,
    kernelPath: run.kernelPath ?? null,
    memorySnapshot: run.memorySnapshot ?? null,
  };
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
  const captureToggle = $('capture-transcript-toggle');
  if (captureToggle) {
    captureToggle.checked = state.captureTranscriptEnabled === true;
    captureToggle.addEventListener('change', (e) => {
      state.captureTranscriptEnabled = e.target.checked === true;
    });
  }
}
