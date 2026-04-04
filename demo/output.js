import { state } from './ui/state.js';

function $(id) { return document.getElementById(id); }

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

export function setPhase(label) {
  setText('output-phase', label);
}

export function setTokSec(value) {
  if (!state.liveTokSec) return;
  setText('output-toks', value != null ? `${value.toFixed(1)} tok/s` : '');
}

export function clearTokSec() {
  setText('output-toks', '');
}

export function setPrefillProgress(percent) {
  const bar = $('output-prefill-bar');
  if (bar) bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

export function appendToken(text) {
  const el = $('output-text');
  if (el) el.textContent += text;
}

export function clearOutput() {
  const el = $('output-text');
  if (el) el.textContent = '';
  setPrefillProgress(0);
  setPhase('');
  clearTokSec();
}

export function showTokenPress(show) {
  const plain = $('output-text');
  const tpOut = $('token-press-output');
  const tpCtrl = $('token-press-controls');
  if (plain) plain.hidden = show;
  if (tpOut) tpOut.hidden = !show;
  if (tpCtrl) tpCtrl.hidden = !show;
}

export function setFinalStats(stats) {
  if (!stats) return;
  const parts = [];
  if (stats.totalTokens != null) parts.push(`${stats.totalTokens} tokens`);
  if (stats.prefillMs != null) parts.push(`prefill ${stats.prefillMs.toFixed(0)}ms`);
  if (stats.decodeMs != null) parts.push(`decode ${stats.decodeMs.toFixed(0)}ms`);
  if (stats.tokPerSec != null) parts.push(`${stats.tokPerSec.toFixed(1)} tok/s`);
  setText('output-toks', parts.join(' · '));
}
