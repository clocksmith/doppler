import {
  getRuntimeConfig,
  captureMemorySnapshot,
  buildReferenceTranscriptSeed,
  resolveExecutionGraphHash,
  captureKvCacheByteProof,
  digestLogitsForTranscript,
} from 'doppler-gpu/tooling';
import { state } from './ui/state.js';
import { getSettings } from './settings.js';
import { getPrompt, setGenerating, setRunEnabled } from './input.js';
import {
  clearOutput,
  clearTokSec,
  setPhase,
  setTokSec,
  appendToken,
  setPrefillProgress,
  showTokenPress,
  setFinalStats,
} from './output.js';
import { setExportEnabled, setTranscriptExportEnabled } from './report.js';
import { createTokenPress } from './ui/token-press/index.js';
import { createTokenPressSession } from './ui/token-press/bridge.js';
import { buildTokenTraceRecord, summarizePerplexityRecords } from './ui/token-press/metrics.js';
import {
  updateXrayPanels,
  isXrayEnabled,
  isXrayProfilingNeeded,
  resetXray,
} from './ui/xray/index.js';

let tokenPress = null;
let tokenPressSession = null;

function $(id) { return document.getElementById(id); }

function setStatus(text, busy) {
  const dot = $('status-dot');
  const txt = $('status-text');
  if (dot) {
    dot.classList.toggle('is-ready', !busy);
    dot.classList.toggle('is-busy', busy);
  }
  if (txt) txt.textContent = text;
}

function resetGenerationTelemetry() {
  state.lastRun = null;
  state.lastInferenceStats = null;
  state.lastMemoryStats = null;
  state.lastReferenceTranscript = null;
  if (isXrayEnabled()) {
    resetXray();
  }
}

function resolvePromptTokenIdsFromPipeline(pipeline, prompt) {
  if (!pipeline?.tokenizer || typeof pipeline.tokenizer.encode !== 'function') {
    return null;
  }
  try {
    const ids = pipeline.tokenizer.encode(prompt);
    if (!Array.isArray(ids)) return null;
    return ids.map((value) => Number(value)).filter((value) => Number.isInteger(value));
  } catch {
    return null;
  }
}

function updateTelemetry(pipeline, overrides = {}) {
  const pipelineStats = pipeline?.getStats?.() ?? null;
  state.lastInferenceStats = {
    ...(pipelineStats || state.lastInferenceStats || {}),
    ...overrides,
  };
  state.lastMemoryStats = pipeline?.getMemoryStats?.() ?? state.lastMemoryStats;
  if (isXrayEnabled()) {
    updateXrayPanels(pipeline);
  }
}

export function onModelLoaded(pipeline, modelId) {
  setRunEnabled(true);
  setStatus('Ready', false);
}

export async function runGeneration() {
  const pipeline = state.pipeline;
  if (!pipeline) return;

  const prompt = getPrompt();
  if (!prompt) return;

  const settings = getSettings();
  // settings.js has already applied the runtime profile and synced runtime
  // fields to the engine. settings contains generation params (temperature,
  // topK, topP, maxTokens) which we use directly in the decode loop.

  // Reset UI
  clearOutput();
  setGenerating(true);
  setStatus('Running...', true);
  setExportEnabled(false);
  state.generating = true;
  state.prefilling = true;
  state.abortController = new AbortController();
  resetGenerationTelemetry();
  clearTokSec();

  const useTokenPress = state.tokenPressEnabled;
  showTokenPress(useTokenPress);

  const tokens = [];

  try {
    if (useTokenPress) {
      await runWithTokenPress(pipeline, prompt, settings, tokens);
    } else {
      await runPlainGeneration(pipeline, prompt, settings, tokens);
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      setPhase(`Error: ${err.message}`);
    }
  } finally {
    state.generating = false;
    state.prefilling = false;
    setGenerating(false);
    setStatus('Ready', false);

    // Capture final stats
    const run = state.lastRun;
    if (run) {
      setFinalStats(run);
      setExportEnabled(true);
    }

    // Update xray if enabled
    if (isXrayEnabled()) {
      try {
        updateXrayPanels(pipeline);
      } catch {
        // xray update is best-effort
      }
    }
  }
}

async function runPlainGeneration(pipeline, prompt, settings, tokens) {
  setPhase('Prefill');
  setPrefillProgress(10);
  const runStart = performance.now();
  const captureEnabled = state.captureTranscriptEnabled === true;
  const captureTokenIds = [];
  const captureLogitsDigests = [];
  const generationOptions = {
    temperature: settings.temperature,
    topK: settings.topK,
    topP: settings.topP,
    maxTokens: settings.maxTokens,
    signal: state.abortController?.signal,
    useChatTemplate: true,
  };
  if (isXrayProfilingNeeded()) {
    generationOptions.profile = true;
  }
  if (captureEnabled) {
    generationOptions.disableCommandBatching = true;
    generationOptions.onToken = (tokenId) => {
      if (Number.isInteger(tokenId)) captureTokenIds.push(tokenId);
    };
    generationOptions.onLogits = (logits, context) => {
      captureLogitsDigests.push(digestLogitsForTranscript(logits, {
        ...context,
        index: captureLogitsDigests.length,
      }));
    };
  }
  let firstTokenAt = 0;
  let stepCount = 0;
  for await (const text of pipeline.generate(prompt, generationOptions)) {
    if (state.abortController?.signal?.aborted) break;
    const now = performance.now();
    if (!firstTokenAt) {
      firstTokenAt = now;
      state.prefilling = false;
      setPrefillProgress(100);
      setPhase('Decode');
    }
    tokens.push({ text });
    appendToken(text);
    stepCount += 1;
    if (state.liveTokSec && firstTokenAt) {
      const elapsed = Math.max(1, now - firstTokenAt);
      setTokSec(stepCount / (elapsed / 1000));
    }
    const prefillMs = firstTokenAt ? Math.max(0, firstTokenAt - runStart) : 0;
    const decodeMs = firstTokenAt ? Math.max(0, now - firstTokenAt) : 0;
    updateTelemetry(pipeline, {
      prefillTimeMs: prefillMs,
      ttftMs: prefillMs,
      decodeTimeMs: decodeMs,
      decodeTokens: stepCount,
      totalTimeMs: prefillMs + decodeMs,
      tokensGenerated: stepCount,
    });
  }
  const runEnd = performance.now();
  if (!firstTokenAt) {
    setPrefillProgress(100);
  }
  const fallbackPrefillMs = firstTokenAt
    ? Math.max(0, firstTokenAt - runStart)
    : Math.max(0, runEnd - runStart);
  const fallbackDecodeMs = firstTokenAt ? Math.max(0, runEnd - firstTokenAt) : 0;
  updateTelemetry(pipeline, {
    prefillTimeMs: fallbackPrefillMs,
    ttftMs: fallbackPrefillMs,
    decodeTimeMs: fallbackDecodeMs,
    decodeTokens: stepCount,
    totalTimeMs: fallbackPrefillMs + fallbackDecodeMs,
    tokensGenerated: stepCount,
  });
  const stats = state.lastInferenceStats || {};
  const prefillMs = Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : fallbackPrefillMs;
  const decodeMs = Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : fallbackDecodeMs;
  const totalTokens = Number.isFinite(stats.tokensGenerated) ? stats.tokensGenerated : stepCount;
  const tokPerSec = decodeMs > 0 ? totalTokens / (decodeMs / 1000) : 0;

  state.lastRun = {
    mode: 'plain',
    prefillMs,
    decodeMs,
    totalTokens,
    tokPerSec,
    tokens,
    config: { ...settings },
    kernelPath: getRuntimeConfig()?.kernelPath ?? null,
    memorySnapshot: captureMemorySnapshot(),
    perplexity: null,
    tokenPress: {
      enabled: false,
      topKSize: 0,
      tooltipRecords: 0,
    },
  };

  if (captureEnabled) {
    await finalizeReferenceTranscript(pipeline, {
      prompt,
      tokenIds: captureTokenIds,
      logitsDigests: captureLogitsDigests,
      output: tokens.map((t) => t.text).join(''),
      prefillMs,
      decodeMs,
      totalTokens,
    });
  }
}

async function finalizeReferenceTranscript(pipeline, run) {
  try {
    const manifest = pipeline?.manifest ?? pipeline?.modelConfig?.manifest ?? null;
    const executionGraphHash = resolveExecutionGraphHash(manifest);
    const kvCacheByteProof = await captureKvCacheByteProof(pipeline, true);
    const promptTokenIds = resolvePromptTokenIdsFromPipeline(pipeline, run.prompt);
    const seedRun = {
      prompt: run.prompt,
      promptInput: run.prompt,
      promptTokenIds,
      tokenIds: run.tokenIds,
      tokenDiagnostics: null,
      logitsDigests: run.logitsDigests,
      kvCacheByteProof,
      output: run.output,
      phase: {
        prefillMs: run.prefillMs,
        decodeMs: run.decodeMs,
        prefillTokens: promptTokenIds ? promptTokenIds.length : null,
        decodeTokens: run.totalTokens,
        stopReason: 'max_tokens_or_eos',
        kvCache: pipeline?.kvCache ?? null,
      },
    };
    const transcript = buildReferenceTranscriptSeed(seedRun, {
      executionGraphHash,
      kvCache: pipeline?.kvCache ?? null,
    });
    state.lastReferenceTranscript = transcript;
    setTranscriptExportEnabled(true);
  } catch (err) {
    state.lastReferenceTranscript = null;
    setTranscriptExportEnabled(false);
    // Capture is best-effort — surface errors through UI rather than crashing the run.
    setPhase(`Transcript capture failed: ${err.message}`);
  }
}

async function runWithTokenPress(pipeline, prompt, settings, tokens) {
  const outputEl = $('token-press-output');
  const controlsEl = $('token-press-controls');
  if (!outputEl || !controlsEl) return;

  // Dispose previous
  if (tokenPress) tokenPress.dispose();
  if (tokenPressSession) tokenPressSession.dispose();

  tokenPress = createTokenPress(outputEl, controlsEl, {
    trailSize: 8,
    topKSize: 10,
  });

  tokenPressSession = createTokenPressSession(pipeline, tokenPress, prompt, {
    temperature: settings.temperature,
    topK: settings.topK,
    topP: settings.topP,
    maxTokens: settings.maxTokens,
    useChatTemplate: true,
  });

  tokenPress.attachSession(tokenPressSession);

  setPhase('Prefill');
  setPrefillProgress(50);
  const prefillStart = performance.now();
  await tokenPressSession.prefill();
  const prefillMs = performance.now() - prefillStart;
  setPrefillProgress(100);
  state.prefilling = false;
  updateTelemetry(pipeline, {
    prefillTimeMs: prefillMs,
    prefillTokens: tokenPressSession.prefillTokenIds.length,
    ttftMs: prefillMs,
    decodeTimeMs: 0,
    decodeTokens: 0,
    totalTimeMs: prefillMs,
    tokensGenerated: 0,
  });

  setPhase('Decode');
  const decodeStart = performance.now();
  tokenPress.play();

  // Wait for generation to finish
  await new Promise((resolve) => {
    let firstTokenAt = 0;
    const check = setInterval(() => {
      if (tokenPressSession.finished || state.abortController?.signal?.aborted) {
        clearInterval(check);
        tokenPress.pause();
        resolve();
        return;
      }
      const currentTokens = tokenPressSession.tokenCount;
      if (currentTokens > 0 && !firstTokenAt) {
        firstTokenAt = performance.now();
      }
      const now = performance.now();
      const decodeMs = Math.max(0, now - decodeStart);
      updateTelemetry(pipeline, {
        prefillTimeMs: prefillMs,
        prefillTokens: tokenPressSession.prefillTokenIds.length,
        ttftMs: firstTokenAt ? Math.max(0, firstTokenAt - decodeStart) : prefillMs,
        decodeTimeMs: decodeMs,
        decodeTokens: currentTokens,
        totalTimeMs: prefillMs + decodeMs,
        tokensGenerated: currentTokens,
      });
    }, 50);
  });

  const decodeMs = performance.now() - decodeStart;
  const totalTokens = tokenPressSession.tokenCount;
  const tokPerSec = totalTokens > 0 ? totalTokens / (decodeMs / 1000) : 0;
  updateTelemetry(pipeline, {
    prefillTimeMs: prefillMs,
    prefillTokens: tokenPressSession.prefillTokenIds.length,
    ttftMs: prefillMs,
    decodeTimeMs: decodeMs,
    decodeTokens: totalTokens,
    totalTimeMs: prefillMs + decodeMs,
    tokensGenerated: totalTokens,
  });
  const tokenTrace = tokenPress.queue.committed.map((record, index) => buildTokenTraceRecord(record, index));
  const perplexity = summarizePerplexityRecords(tokenTrace);

  state.lastRun = {
    mode: 'token-press',
    prefillMs,
    decodeMs,
    totalTokens,
    tokPerSec,
    tokens: tokenTrace,
    config: { ...settings },
    kernelPath: getRuntimeConfig()?.kernelPath ?? null,
    memorySnapshot: captureMemorySnapshot(),
    perplexity,
    tokenPress: {
      enabled: true,
      topKSize: 10,
      tooltipRecords: tokenTrace.length,
    },
  };
}

export function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
  }
}
