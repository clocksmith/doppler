import { state } from './ui/state.js';
import { getSettings } from './settings.js';
import {
  buildConversationRequest,
  clearPrompt,
  getPrompt,
  recordConversationTurn,
  resetConversationForModel,
  setGenerating,
  syncSendButton,
} from './input.js';
import {
  beginChatTurn,
  clearTokSec,
  renderWordQuality,
  setFinalStats,
  setPhase,
  setPrefillProgress,
  showWordQuality,
} from './output.js';
import { setExportEnabled } from './report.js';
import {
  isXrayEnabled,
  resetXray,
  updateXrayPanels,
} from './ui/xray/index.js';

function $(id) {
  return document.getElementById(id);
}

function setStatus(text, busy) {
  const dot = $('status-dot');
  const label = $('status-text');
  dot?.classList.toggle('is-ready', !busy);
  dot?.classList.toggle('is-busy', busy);
  if (label) label.textContent = text;
}

function resolvePolicyId() {
  if (isXrayEnabled()) return 'demo/deep-xray';
  if (state.wordQualityEnabled) return 'demo/guided-quality';
  return 'demo/always-on';
}

export function onModelLoaded(model, modelId) {
  state.model = model;
  resetConversationForModel(modelId);
  syncSendButton();
  setStatus('Ready', false);
}

export async function runGeneration() {
  const model = state.model;
  const prompt = getPrompt();
  if (!model || !prompt) return;

  const settings = getSettings();
  const conversationRequest = buildConversationRequest(prompt);
  const policyId = resolvePolicyId();
  beginChatTurn(conversationRequest.messages);
  clearPrompt();
  clearTokSec();
  setPrefillProgress(10);
  setPhase('Running');
  setGenerating(true);
  setStatus('Running…', true);
  setExportEnabled(false);
  state.generating = true;
  state.prefilling = true;
  state.abortController = new AbortController();
  resetXray();

  try {
    const receipt = await model.inspect.generate(conversationRequest.currentPrompt, {
      policyId,
      topKSize: 8,
      generation: {
        temperature: settings.temperature,
        topK: settings.topK,
        topP: settings.topP,
        maxTokens: settings.maxTokens,
        signal: state.abortController.signal,
        useChatTemplate: true,
      },
    });
    const qualityEnabled = receipt.quality != null;
    showWordQuality(qualityEnabled);
    if (qualityEnabled) {
      renderWordQuality(receipt.quality);
    } else {
      const output = $('output-text');
      if (output) output.textContent = receipt.outputText;
    }
    const stats = receipt.generationEvidence?.stats ?? {};
    const totalTokens = receipt.generatedTokenIds.length;
    const totalMs = receipt.wallTimingMs;
    const prefillMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null;
    const decodeMs = Number.isFinite(stats.decodeTimeMs)
      ? stats.decodeTimeMs
      : (prefillMs == null ? totalMs : Math.max(0, totalMs - prefillMs));
    const tokPerSec = decodeMs > 0 ? totalTokens / (decodeMs / 1000) : null;
    state.lastInspection = receipt;
    globalThis.__DOPPLER_DEMO_EVIDENCE__ = receipt;
    state.lastInferenceStats = stats;
    state.lastRun = {
      mode: qualityEnabled ? 'guided-quality' : 'always-on',
      output: receipt.outputText,
      tokens: receipt.tokens,
      totalTokens,
      prefillMs,
      decodeMs,
      tokPerSec,
      prompt: conversationRequest.currentPrompt,
      promptInput: conversationRequest.promptInput,
      config: { ...settings },
      observationPolicy: receipt.policy,
      comparisonFingerprint: receipt.fingerprint,
      perplexity: receipt.quality,
      wordQuality: {
        enabled: qualityEnabled,
        topKSize: qualityEnabled ? 8 : 0,
        tooltipRecords: receipt.tokens.length,
      },
    };
    recordConversationTurn(conversationRequest, receipt.outputText);
    if (qualityEnabled) {
      beginChatTurn(state.conversationHistory.slice(0, -1));
      showWordQuality(true);
      renderWordQuality(receipt.quality);
    }
    updateXrayPanels(receipt);
    setFinalStats(state.lastRun);
    setExportEnabled(true);
    setPhase(receipt.performanceRepresentative ? 'Complete · representative wall timing' : 'Complete · diagnostic timing');
  } catch (error) {
    if (error?.name !== 'AbortError') {
      setPhase(`Error: ${error?.message ?? error}`);
    }
  } finally {
    state.generating = false;
    state.prefilling = false;
    state.abortController = null;
    setGenerating(false);
    setPrefillProgress(100);
    setStatus('Ready', false);
  }
}

export function stopGeneration() {
  state.abortController?.abort();
}
