import { getRuntimeConfig, captureMemorySnapshot } from 'doppler-gpu';
import { sample } from '../src/inference/pipelines/text/sampling.js';
import { state } from './ui/state.js';
import { getSettings } from './settings.js';
import { getPrompt, getImage, setGenerating, setRunEnabled } from './input.js';
import {
  clearOutput,
  setPhase,
  setTokSec,
  appendToken,
  setPrefillProgress,
  showTokenPress,
  setFinalStats,
} from './output.js';
import { setExportEnabled } from './report.js';
import { createTokenPress } from './ui/token-press/index.js';
import { createTokenPressSession } from './ui/token-press/bridge.js';
import { updateXrayPanels, isXrayEnabled } from './ui/xray/index.js';

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

  const useTokenPress = state.tokenPressEnabled;
  showTokenPress(useTokenPress);

  const tokens = [];
  let prefillMs = 0;
  let decodeStartTime = 0;

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
  // Prefill
  setPhase('Prefill');
  setPrefillProgress(50);
  const prefillStart = performance.now();
  const prefillResult = await pipeline.prefillWithLogits(prompt, {
    useChatTemplate: true,
  });
  const prefillMs = performance.now() - prefillStart;
  setPrefillProgress(100);

  // Decode
  setPhase('Decode');
  const decodeStart = performance.now();
  let logits = prefillResult.logits;
  const generatedIds = [...(prefillResult.tokens ?? [])];
  const stopTokenIds = pipeline.modelConfig?.stopTokenIds ?? [];
  const maxTokens = settings.maxTokens || 256;
  let stepCount = 0;

  while (stepCount < maxTokens) {
    if (state.abortController?.signal?.aborted) break;

    const tokenId = sampleToken(logits, settings);
    if (stopTokenIds.includes(tokenId)) break;

    generatedIds.push(tokenId);
    const text = decodeToken(pipeline, [tokenId]);
    tokens.push({ tokenId, text });
    appendToken(text);
    stepCount++;

    // Live tok/s
    if (state.liveTokSec && stepCount % 4 === 0) {
      const elapsed = performance.now() - decodeStart;
      setTokSec(stepCount / (elapsed / 1000));
    }

    // Next step
    try {
      const result = await pipeline.decodeStepLogits([tokenId]);
      logits = result.logits;
    } catch {
      break;
    }
  }

  const decodeMs = performance.now() - decodeStart;
  const tokPerSec = stepCount > 0 ? stepCount / (decodeMs / 1000) : 0;

  state.lastRun = {
    prefillMs,
    decodeMs,
    totalTokens: stepCount,
    tokPerSec,
    tokens,
    config: { ...settings },
    kernelPath: getRuntimeConfig()?.kernelPath ?? null,
    memorySnapshot: captureMemorySnapshot(),
  };
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

  setPhase('Decode');
  const decodeStart = performance.now();
  tokenPress.play();

  // Wait for generation to finish
  await new Promise((resolve) => {
    const check = setInterval(() => {
      if (tokenPressSession.finished || state.abortController?.signal?.aborted) {
        clearInterval(check);
        tokenPress.pause();
        resolve();
      }
    }, 50);
  });

  const decodeMs = performance.now() - decodeStart;
  const totalTokens = tokenPressSession.tokenCount;
  const tokPerSec = totalTokens > 0 ? totalTokens / (decodeMs / 1000) : 0;

  state.lastRun = {
    prefillMs,
    decodeMs,
    totalTokens,
    tokPerSec,
    tokens,
    config: { ...settings },
    kernelPath: getRuntimeConfig()?.kernelPath ?? null,
    memorySnapshot: captureMemorySnapshot(),
  };
}

function sampleToken(logits, settings) {
  if (!logits || logits.length === 0) return 0;

  const temp = settings.temperature ?? 0;
  if (temp === 0) {
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  return sample(Float32Array.from(logits), {
    temperature: temp,
    topP: settings.topP ?? 1.0,
    topK: settings.topK ?? 0,
  });
}

function decodeToken(pipeline, ids) {
  try {
    return pipeline.tokenizer?.decode?.(ids, true, false) ?? `[${ids[0]}]`;
  } catch {
    return `[${ids[0]}]`;
  }
}

export function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
  }
}
