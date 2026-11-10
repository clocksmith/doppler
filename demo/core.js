import { SAMPLE_INSPECTION_RECEIPT } from './data/sample-inspection.js';
import { state } from './ui/state.js';
import { isActivationRecordingEnabled, beginActivationRecording, acceptActivationFrame,
  finishActivationRecording, clearActivationRecording, getActivationRecording, importActivationRecording } from './ui/activation-grid/index.js';
import { SHELL_MANIFEST_DIGEST } from './generated-shell-manifest.js';
import activityConfig from './ui/activation-grid/config.json' with { type: 'json' };
import { syncModelControls } from './models.js';
import { getSettings } from './settings.js';
import {
  buildConversationRequest,
  clearPrompt,
  clearConversationHistory,
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
import { selectToken, setTokenInspectorOptions, updateTokenInspector } from './ui/token-inspector/index.js';
import {
  isXrayEnabled,
  resetXray,
  updateXrayPanels,
} from './ui/xray/index.js';

function $(id) {
  return document.getElementById(id);
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
  setPhase('Running', { busy: true, quiet: true });
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
  beginActivationRecording(prompt);

  try {
    const receipt = await model.inspect.generate(conversationRequest.promptInput, {
      policyId,
      topKSize: 8,
      onEvent(event) {
        if (receiving && event.type === 'activation-frame') { acceptActivationFrame(event); return; }
        if (!receiving || event.type !== 'token' || signal.aborted) return;
        if (state.prefilling) {
          state.prefilling = false;
          setPhase('Generating', { busy: true, quiet: true });
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
    setPhase('Complete', { quiet: true });
  } catch (error) {
    receiving = false;
    const partialOutput = stream.finish();
    finishActivationRecording(signal.aborted ? 'stopped' : 'error', null, partialOutput);
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
  const trace = run?.receipt.activationTrace;
  if (trace && getActivationRecording()?.trace !== trace) {
    importActivationRecording({ schema: 'doppler.activation-recording/v1', demoBuild: SHELL_MANIFEST_DIGEST,
      display: { layout: activityConfig.layout, palette: activityConfig.palette, scale: activityConfig.scale },
      trace, prompt: run.prompt, outputText: run.receipt.outputText, fingerprint: run.receipt.fingerprint,
      inputTokenIds: run.receipt.inputTokenIds, generatedTokenIds: run.receipt.generatedTokenIds });
  } else if (!trace) clearActivationRecording();
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
  setPhase('Showing branch', { quiet: true });
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
    setPhase('Showing existing branch', { quiet: true });
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
  setPhase('Branching', { busy: true, quiet: true });
  beginActivationRecording(state.branchPrompt);
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
          setPhase('Generating branch', { busy: true, quiet: true });
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
    setPhase('Branch complete', { quiet: true });
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
    if (tree === state.branchTree) {
      renderActiveBranch();
      publishActiveRun();
    }
  }
}

// Opening activity replaces the displayed exchange as one operation. Old files
// without prompt text remain inspectable, but never borrow the current draft.
export function showRecordedActivity(recording, { preserveDraft = false } = {}) {
  if (state.generating || state.modelBusy || state.settingsBusy) throw new Error('Wait for the current run to finish before opening a recording.');
  if (matchesSample(recording)) {
    loadSampleInspection(SAMPLE_INSPECTION_RECEIPT, { preserveDraft });
    return;
  }
  clearConversationHistory();
  beginChatTurn(recording.prompt ? [{ role: 'user', content: recording.prompt }] : []);
  setAssistantText(recording.outputText ?? '');
  if (!preserveDraft) setPromptValue(recording.prompt ?? '');
  const note = $('recorded-run-note');
  if (note) {
    note.textContent = `Recorded run · ${recording.trace.identity.logicalModelId ?? recording.trace.identity.modelId}`
      + (recording.prompt ? '' : ' · prompt text not saved')
      + (recording.outputText == null ? ' · answer text not saved' : '');
    note.hidden = false;
  }
  resetXray();
  setPhase('Recorded activity', { quiet: true });
}

export function loadSampleInspection(receipt, { preserveDraft = false } = {}) {
  if (state.generating || state.modelBusy || state.settingsBusy) return;
  clearConversationHistory();
  if (!preserveDraft) setPromptValue(receipt.prompt);
  else if (getPrompt() === receipt.prompt) clearPrompt();
  beginChatTurn([{ role: 'user', content: receipt.prompt }]);
  syncBranchView();
  const promptLabel = document.querySelector('#chat-thread .chat-role');
  if (promptLabel) promptLabel.textContent = 'Prompt';
  const liveMessage = $('live-assistant-message');
  const outputText = $('output-text');
  if (outputText) outputText.textContent = receipt.outputText;
  const note = $('recorded-run-note');
  if (note) {
    // Named from the saved run, never from the model picker above it.
    const recordedModelId = receipt.recording?.modelId ?? '';
    const recordedModel = state.quickModelCatalog?.find((entry) => entry.modelId === recordedModelId)?.label ?? recordedModelId;
    note.textContent = `Recorded replay · ${recordedModel} · not running on this device`;
    note.title = `${recordedModelId} · ${receipt.recording?.publishedAt ?? ''}`;
    note.hidden = false;
  }
  const details = $('recording-details');
  if (details) {
    details.textContent = `${receipt.recording?.modelId} · published ${receipt.recording?.publishedAt}. Text, probabilities and activity are from the same saved run. Playback pairs each completed layer stack with its predicted token, not original elapsed time. No model runs for this replay.`;
    details.hidden = false;
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
  setPhase('Recorded answer', { quiet: true });
  clearTokSec();
  setGenerating(false);
}

// Words of the shown answer that the model gave less than even odds, by the
// token position they were predicted at.
export function unsureWords() {
  const receipt = state.lastInspection;
  if (!Array.isArray(receipt?.tokens) || !Array.isArray(receipt?.inputTokenIds)) return [];
  const firstPrediction = receipt.inputTokenIds.length - 1;
  return receipt.tokens.flatMap((token, index) => (
    Number.isFinite(token?.probability) && token.probability < activityConfig.unsureBelowProbability
      ? [{ position: firstPrediction + index, probability: token.probability }] : []));
}

// Shows what else the model considered for the word predicted at a token position.
export function showWordAt(position) {
  const receipt = state.lastInspection;
  if (!Array.isArray(receipt?.tokens) || !Array.isArray(receipt?.inputTokenIds)) return;
  const index = position - (receipt.inputTokenIds.length - 1);
  if (index >= 0 && index < receipt.tokens.length) selectToken(index);
}

// Only a matching receipt may supply decoded text or probabilities for a trace.
function matchesSample(recording) {
  return recording.prompt === SAMPLE_INSPECTION_RECEIPT.prompt
    && `sha256:${recording.trace.identity.manifestHash?.replace(/^sha256:/, '')}` === SAMPLE_INSPECTION_RECEIPT.fingerprint.identity.artifact.manifestHash
    && recording.fingerprint?.fullDigest === SAMPLE_INSPECTION_RECEIPT.fingerprint.fullDigest
    && recording.outputText === SAMPLE_INSPECTION_RECEIPT.outputText
    && JSON.stringify(recording.inputTokenIds) === JSON.stringify(SAMPLE_INSPECTION_RECEIPT.inputTokenIds)
    && JSON.stringify(recording.generatedTokenIds) === JSON.stringify(SAMPLE_INSPECTION_RECEIPT.generatedTokenIds);
}

// The grid owns the clock. Each completed layer stack identifies a prediction;
// scrubbing backwards removes later words as well as later numerical activity.
export function updateRecordedActivity(recording, point) {
  if (!matchesSample(recording) || state.lastInspection !== SAMPLE_INSPECTION_RECEIPT) return;
  const firstPrediction = SAMPLE_INSPECTION_RECEIPT.inputTokenIds.length - 1;
  const boundaries = recording.trace.frames.filter(frame => frame.status === 'captured'
    && frame.layer === recording.trace.geometry.layers - 1);
  const count = boundaries.filter(frame => {
    const position = frame.tokenStart + frame.tokenIds.length - 1;
    return position >= firstPrediction && position <= point.position && frame.atMs <= point.atMs;
  }).length;
  const tokens = SAMPLE_INSPECTION_RECEIPT.tokens.slice(0, count);
  const text = tokens.map(token => token.text).join('');
  if ($('output-text')?.textContent === text) return;
  setAssistantText(text);
  renderWordQuality(count === SAMPLE_INSPECTION_RECEIPT.tokens.length ? SAMPLE_INSPECTION_RECEIPT.quality : null, text);
  renderTokenInspection(tokens);
  showTokenInspectorView(tokens.length > 0);
}
