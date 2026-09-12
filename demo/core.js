import { state } from './ui/state.js';
import { syncModelControls } from './models.js';
import { getSettings } from './settings.js';
import {
  buildConversationRequest,
  clearPrompt,
  getPrompt,
  recordConversationTurn,
  setPromptValue,
  resetConversationForModel,
  setGenerating,
  syncSendButton,
} from './input.js';
import {
  beginChatTurn,
  clearTokSec,
  createOutputStream,
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
  if (!model || !prompt || state.generating || state.modelBusy || state.settingsBusy) return;

  let settings;
  try {
    settings = getSettings();
  } catch (error) {
    setPhase(error.message);
    return;
  }
  const conversationRequest = buildConversationRequest(prompt);
  const policyId = resolvePolicyId();
  beginChatTurn(conversationRequest.messages);
  clearPrompt();
  clearTokSec();
  setPrefillProgress(10);
  setPhase('Running');
  setStatus('Running…', true);
  setExportEnabled(false);
  state.generating = true;
  state.prefilling = true;
  state.lastImportedReport = null;
  state.lastRun = null;
  state.lastInspection = null;
  delete globalThis.__DOPPLER_DEMO_EVIDENCE__;
  setGenerating(true);
  syncModelControls();
  state.abortController = new AbortController();
  const signal = state.abortController.signal;
  const stream = createOutputStream((ids) => model.advanced.decodeTokenIds(ids), signal);
  let receiving = true;
  resetXray();

  try {
    const receipt = await model.inspect.generate(conversationRequest.currentPrompt, {
      policyId,
      topKSize: 8,
      onEvent(event) {
        if (!receiving || event.type !== 'token' || signal.aborted) return;
        if (state.prefilling) {
          state.prefilling = false;
          setPhase('Generating');
          setPrefillProgress(100);
        }
        stream.push(event.tokenId);
      },
      generation: {
        temperature: settings.temperature,
        topK: settings.topK,
        topP: settings.topP,
        maxTokens: settings.maxTokens,
        signal,
        useChatTemplate: true,
      },
    });
    receiving = false;
    signal.throwIfAborted();
    stream.finish(receipt.outputText);
    const qualityEnabled = receipt.quality != null;
    showWordQuality(qualityEnabled);
    if (qualityEnabled) {
      renderWordQuality(receipt.quality);
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
    recordConversationTurn(conversationRequest, receipt.outputText, { render: false });
    updateXrayPanels(receipt);
    setFinalStats(state.lastRun);
    setExportEnabled(true);
    setPhase(receipt.performanceRepresentative ? 'Complete' : 'Complete · diagnostic timing');
  } catch (error) {
    receiving = false;
    const partialOutput = stream.finish();
    if (partialOutput) recordConversationTurn(conversationRequest, partialOutput, { render: false });
    setPhase(error?.name === 'AbortError' ? 'Stopped' : `Error: ${error?.message ?? error}`);
    if (!getPrompt()) setPromptValue(prompt);
  } finally {
    receiving = false;
    stream.finish();
    state.generating = false;
    state.prefilling = false;
    state.abortController = null;
    setGenerating(false);
    syncModelControls();
    setPrefillProgress(100);
    setStatus('Ready', false);
  }
}

export function stopGeneration() {
  state.abortController?.abort();
}
