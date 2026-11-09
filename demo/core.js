import { state } from './ui/state.js';
import { isActivationRecordingEnabled, beginActivationRecording, acceptActivationFrame,
  finishActivationRecording } from './ui/activation-grid/index.js';
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
  renderTokenInspection,
  renderWordQuality,
  setFinalStats,
  setPhase,
  setPrefillProgress,
  setAssistantText,
  showTokenInspectorView,
  showWordQuality,
} from './output.js';
import { setExportEnabled } from './report.js';
import {
  activeBranch,
  appendBranchToken,
  branchInputTokenIds,
  createBranchTree,
  forkGroups,
  removeBranch,
  selectBranch,
  settleBranch,
  startBranch,
} from './branches.js';
import { setTokenInspectorOptions, updateTokenInspector } from './ui/token-inspector/index.js';
import {
  isXrayEnabled,
  resetXray,
  updateXrayPanels,
} from './ui/xray/index.js';

function $(id) {
  return document.getElementById(id);
}

function setStatus(text, busy) {
  const status = $('status');
  const dot = $('status-dot');
  const label = $('status-text');
  dot?.classList.toggle('is-ready', !busy);
  dot?.classList.toggle('is-busy', busy);
  if (label) label.textContent = text;
  if (status) status.hidden = busy;
}

// One owner for the run state a completed receipt publishes (answer or branch).
function recordInspectionRun(receipt, { prompt, promptInput, config, qualityEnabled = false }) {
  const stats = receipt.generationEvidence?.stats ?? {};
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : null;
  state.lastInspection = receipt;
  globalThis.__DOPPLER_DEMO_EVIDENCE__ = receipt;
  state.lastInferenceStats = stats;
  state.lastRun = {
    mode: receipt.policy.id,
    output: receipt.outputText,
    tokens: receipt.tokens,
    totalTokens: receipt.generatedTokenIds.length,
    prefillMs: Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : null,
    decodeMs,
    tokPerSec: Number.isFinite(stats.tokensPerSecond) ? stats.tokensPerSecond
      : Number.isFinite(stats.decodeTokens) && decodeMs > 0 ? stats.decodeTokens / (decodeMs / 1000) : null,
    prompt,
    promptInput,
    config,
    observationPolicy: receipt.policy,
    comparisonFingerprint: receipt.fingerprint,
    perplexity: receipt.quality,
    wordQuality: {
      enabled: qualityEnabled,
      topKSize: qualityEnabled ? 8 : 0,
      tooltipRecords: receipt.tokens.length,
    },
  };
  updateXrayPanels(receipt);
  setFinalStats(state.lastRun);
  setExportEnabled(true);
}

// Every live answer records probabilities and alternatives for branching.
function resolvePolicyId() {
  if (isActivationRecordingEnabled()) return isXrayEnabled() ? 'demo/model-activity-timing' : 'demo/model-activity';
  return isXrayEnabled() ? 'demo/deep-xray' : 'demo/guided-quality';
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
  beginActivationRecording();

  try {
    const receipt = await model.inspect.generate(conversationRequest.promptInput, {
      policyId,
      topKSize: 8,
      onEvent(event) {
        if (receiving && event.type === 'activation-frame') { acceptActivationFrame(event); return; }
        if (!receiving || event.type !== 'token' || signal.aborted) return;
        if (state.prefilling) {
          state.prefilling = false;
          setPhase('Generating');
          setPrefillProgress(100);
        }
        stream.push(event.tokenId, event.token ?? null);
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
    finishActivationRecording('complete', receipt);
    const qualityEnabled = state.wordQualityEnabled && receipt.quality != null;
    showWordQuality(qualityEnabled);
    if (qualityEnabled) {
      renderWordQuality(receipt.quality, receipt.outputText);
    }
    if (Array.isArray(receipt.tokens) && receipt.tokens.length > 0) {
      renderTokenInspection(receipt.tokens);
      showTokenInspectorView(true);
    }
    const run = {
      receipt,
      prompt: conversationRequest.currentPrompt,
      promptInput: conversationRequest.promptInput,
      config: { ...settings },
      qualityEnabled,
    };
    recordInspectionRun(receipt, run);
    recordConversationTurn(conversationRequest, receipt.outputText, { render: false, quality: receipt.quality });
    startBranchTree(run);
    setPhase('Complete');
  } catch (error) {
    finishActivationRecording(signal.aborted ? 'stopped' : 'error');
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
    // The tree was built while the run still counted as generating.
    if (state.branchTree) syncBranchView();
  }
}

export function stopGeneration() {
  state.abortController?.abort();
}

// ---------------------------------------------------------------------------
// Branch: generate normally, then continue from any recorded alternative.
// Each branch replays receipt.inputTokenIds plus the exact answer prefix and
// the chosen token, so attention and recurrent state are rebuilt from tokens,
// never from displayed text.

function startBranchTree(run) {
  const { receipt, prompt } = run;
  state.branchTree = null;
  const records = Array.isArray(receipt.tokens) ? receipt.tokens : [];
  if (!Array.isArray(receipt.inputTokenIds)
    || records.length === 0 || records.length !== receipt.generatedTokenIds.length) {
    syncBranchView();
    return;
  }
  state.branchTree = createBranchTree({
    inputTokenIds: receipt.inputTokenIds,
    tokenIds: receipt.generatedTokenIds,
    records,
  });
  // Each branch keeps the completed run that produced it, so Export and X-Ray
  // follow the branch on screen.
  activeBranch(state.branchTree).run = run;
  state.branchPrompt = prompt;
  state.branchModelId = state.modelId;
  renderActiveBranch({ updateText: false });
}

function branchUnavailableReason() {
  if (!state.branchTree) return 'Only a live answer can branch. Recorded and illustrative examples cannot.';
  if (state.generating || state.modelBusy || state.settingsBusy) return 'Wait for the current run to finish.';
  return 'Load the model that wrote this answer to branch from it.';
}

export function syncBranchView() {
  const tree = state.branchTree;
  const model = state.model;
  const branchable = Boolean(tree && model && state.branchModelId === state.modelId
    && !state.generating && !state.modelBusy && !state.settingsBusy);
  setTokenInspectorOptions({
    decodeTokenIds: model ? (ids) => model.advanced.decodeTokenIds(ids) : null,
    forks: tree ? forkGroups(tree) : new Map(),
    branchable,
    unavailableReason: branchUnavailableReason(),
    onBranch: (index, tokenId) => { void runBranch(index, tokenId); },
    onSwitch: (id) => switchBranch(id),
  });
}

function updateConversationAnswer(text) {
  const last = state.conversationHistory.length - 1;
  if (last < 0 || state.conversationHistory[last].role !== 'assistant' || !text.trim()) return;
  state.conversationHistory[last] = { role: 'assistant', content: text.trim() };
  if (state.conversationAnnotations[last]) {
    state.conversationAnnotations[last] = { text: text.trim(), quality: null };
  }
}

function renderActiveBranch({ updateText = true } = {}) {
  const tree = state.branchTree;
  if (!tree) return;
  const node = activeBranch(tree);
  updateTokenInspector(node.records);
  syncBranchView();
  if (updateText && state.model) {
    const text = state.model.advanced.decodeTokenIds(node.pathIds);
    setAssistantText(text);
    updateConversationAnswer(text);
  }
}

// Publishes the shown branch's completed receipt; a stopped branch has none.
function publishActiveRun() {
  const run = state.branchTree ? activeBranch(state.branchTree).run : null;
  if (run) {
    recordInspectionRun(run.receipt, run);
    return;
  }
  state.lastInspection = null;
  state.lastRun = null;
  delete globalThis.__DOPPLER_DEMO_EVIDENCE__;
  resetXray();
  setExportEnabled(false);
}

export function switchBranch(id) {
  if (!state.branchTree || state.generating) return;
  selectBranch(state.branchTree, id);
  renderActiveBranch();
  publishActiveRun();
  setPhase('Showing branch');
}

export async function runBranch(forkIndex, tokenId) {
  const tree = state.branchTree;
  const model = state.model;
  if (!tree || !model || state.branchModelId !== state.modelId
    || state.generating || state.modelBusy || state.settingsBusy) return;
  let settings;
  try {
    settings = getSettings();
  } catch (error) {
    setPhase(error.message);
    return;
  }
  const { node, created } = startBranch(tree, forkIndex, tokenId);
  if (!created) {
    renderActiveBranch();
    publishActiveRun();
    setPhase('Showing existing branch');
    return;
  }
  const remaining = settings.maxTokens - node.pathIds.length;
  if (model.advanced.getStopTokenIds().includes(tokenId) || remaining < 1) {
    settleBranch(node, 'complete');
    renderActiveBranch();
    publishActiveRun();
    setPhase(remaining < 1 ? 'Branch reached the maximum length' : 'Branch ends here');
    return;
  }
  const inputTokenIds = branchInputTokenIds(tree, node);
  state.generating = true;
  state.prefilling = true;
  state.abortController = new AbortController();
  const signal = state.abortController.signal;
  setGenerating(true);
  syncModelControls();
  setExportEnabled(false);
  resetXray();
  setPhase('Branching');
  beginActivationRecording();
  setStatus('Running…', true);
  renderActiveBranch({ updateText: false });
  try {
    const receipt = await model.inspect.generate(state.branchPrompt, {
      policyId: resolvePolicyId(),
      topKSize: 8,
      inputTokenIds,
      onEvent(event) {
        if (event.type === 'activation-frame' && tree === state.branchTree) { acceptActivationFrame(event); return; }
        if (event.type !== 'token' || signal.aborted || tree !== state.branchTree) return;
        if (state.prefilling) {
          state.prefilling = false;
          setPhase('Generating branch');
        }
        appendBranchToken(node, event.tokenId, event.token ?? null);
        updateTokenInspector(node.records);
      },
      generation: {
        temperature: settings.temperature,
        topK: settings.topK,
        topP: settings.topP,
        maxTokens: remaining,
        signal,
        useChatTemplate: true,
      },
    });
    signal.throwIfAborted();
    const records = receipt.tokens.length === receipt.generatedTokenIds.length ? receipt.tokens : null;
    finishActivationRecording('complete', receipt);
    settleBranch(node, 'complete', records ? receipt.generatedTokenIds : null, records);
    node.run = { receipt, prompt: state.branchPrompt, promptInput: null, config: { ...settings } };
    setPhase('Branch complete');
  } catch (error) {
    if (signal.aborted || error?.name === 'AbortError') {
      finishActivationRecording('stopped');
      settleBranch(node, 'stopped');
      setPhase('Branch stopped');
    } else if (tree.nodes.has(node.id)) {
      finishActivationRecording('error');
      removeBranch(tree, node);
      setPhase(`Branch failed: ${error?.message ?? error}`);
    }
  } finally {
    state.generating = false;
    state.prefilling = false;
    state.abortController = null;
    setGenerating(false);
    syncModelControls();
    setStatus('Ready', false);
    if (tree === state.branchTree) {
      renderActiveBranch();
      publishActiveRun();
    }
  }
}

export function loadIllustrativeExample(example) {
  if (!example || state.generating || state.modelBusy || state.settingsBusy) return;
  beginChatTurn([{ role: 'user', content: example.prompt }]);
  const output = $('output-text');
  if (output) output.textContent = example.answer;
  const message = $('live-assistant-message');
  if (message) { message.hidden = false; message.dataset.example = 'illustrative'; }
  const note = $('recorded-run-note');
  if (note) { note.textContent = 'Illustrative answer · not generated by a model.'; note.hidden = false; }
  const recorded = $('inspect-recorded-run');
  if (recorded) recorded.hidden = false;
  state.lastInspection = null;state.lastRun = null;state.lastInferenceStats = null;
  delete globalThis.__DOPPLER_DEMO_EVIDENCE__;
  setExportEnabled(false);clearTokSec();setPhase('Example');
}

export function loadSampleInspection(receipt) {
  if (state.generating || state.modelBusy || state.settingsBusy) return;
  beginChatTurn([{ role: 'user', content: receipt.prompt }]);
  syncBranchView();
  const promptLabel = document.querySelector('#chat-thread .chat-role');
  if (promptLabel) promptLabel.textContent = 'Prompt';
  const liveMessage = $('live-assistant-message');
  const outputText = $('output-text');
  if (outputText) outputText.textContent = receipt.outputText;
  const note = $('recorded-run-note');
  if (note) {
    note.textContent = `Recorded ${receipt.recording?.recordedAt || 'earlier'}, not run on your device. The answer is shown unedited and contains a factual error: “Rayleigh-induced” should be “Rayleigh scattering”. Its likelihoods and times come from that recording.`;
    note.hidden = false;
  }
  if (liveMessage) {
    liveMessage.hidden = false;
    liveMessage.dataset.example = 'true';
  }
  renderWordQuality(receipt.quality, receipt.outputText);
  showWordQuality(state.wordQualityEnabled && receipt.quality != null);

  if (Array.isArray(receipt.tokens) && receipt.tokens.length > 0) {
    renderTokenInspection(receipt.tokens);
    showTokenInspectorView(true);
  }

  state.lastInspection = receipt;
  globalThis.__DOPPLER_DEMO_EVIDENCE__ = receipt;
  const stats = receipt.generationEvidence?.stats ?? {};
  const totalTokens = receipt.generatedTokenIds.length;
  const decodeMs = stats.decodeTimeMs ?? null;
  const prefillMs = stats.prefillTimeMs ?? null;
  const tokPerSec = stats.tokensPerSecond ?? null;
  state.lastInferenceStats = stats;
  state.lastRun = {
    mode: 'sample-inspection',
    output: receipt.outputText,
    tokens: receipt.tokens,
    totalTokens,
    prefillMs,
    decodeMs,
    tokPerSec,
    prompt: receipt.prompt,
    promptInput: receipt.prompt,
    config: {},
    observationPolicy: receipt.policy,
    comparisonFingerprint: receipt.fingerprint,
    perplexity: receipt.quality,
    wordQuality: {
      enabled: true,
      topKSize: 5,
      tooltipRecords: receipt.tokens.length,
    },
  };
  updateXrayPanels(receipt);
  setFinalStats(state.lastRun);
  setExportEnabled(true);
  setPhase('Recorded answer');
  clearTokSec();
  setStatus(state.model ? 'Ready' : 'No model loaded', false);
  setGenerating(false);
}
