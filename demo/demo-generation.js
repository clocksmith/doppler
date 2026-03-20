
function buildRunGenerateOptions() {
  if (state.uiMode === 'embedding') {
    return {};
  }
  const isTranslateMode = state.uiMode === 'translate';
  const temperature = readOptionalNumber($('temperature-input'));
  const topP = readOptionalNumber($('top-p-input'));
  const topK = readOptionalNumber($('top-k-input'), { integer: true });
  const maxTokens = readOptionalNumber($('max-tokens-input'), { integer: true });
  const options = {};
  if (temperature != null) {
    options.temperature = Math.max(0, temperature);
  }
  if (topP != null) {
    options.topP = Math.max(0, Math.min(1, topP));
  }
  if (topK != null) {
    options.topK = Math.max(0, topK);
  }
  if (maxTokens != null && maxTokens > 0) {
    options.maxTokens = Math.max(1, maxTokens);
  }
  if (isTranslateMode) {
    if (temperature == null) {
      options.temperature = DEFAULT_TRANSLATE_TEMPERATURE;
    }
    if (topP == null) {
      options.topP = DEFAULT_TRANSLATE_TOP_P;
    }
    if (topK == null) {
      options.topK = DEFAULT_TRANSLATE_TOP_K;
    }
    if (maxTokens == null) {
      options.maxTokens = DEFAULT_TRANSLATE_MAX_TOKENS;
    }
  } else if (maxTokens == null) {
    options.maxTokens = DEFAULT_TEXT_MAX_TOKENS;
  }
  return options;
}

async function loadPipelineFromStorage(modelId) {
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage');
  }
  const manifest = parseManifest(manifestText);
  await initDevice();
  const device = getDevice();
  return createPipeline(manifest, {
    gpu: { device },
    runtimeConfig: getRuntimeConfig(),
    onProgress: (progress) => updateProgressFromLoader(progress),
  });
}

// Shared pipeline loader — handles overlay, manifest parse, GPU upload, memory stats.
// Callers set their own loading flag before calling and clear it in their own finally.
async function ensurePipeline(modelId, overlayTitle, modeKey) {
  if (!modelId) throw new Error('Select a model before generating');
  if (state.activePipeline && state.activeModelId === modelId) return state.activePipeline;
  if (state.activePipeline) await unloadActivePipeline();
  showProgressOverlay(overlayTitle, modelId);
  try {
    const pipeline = await loadPipelineFromStorage(modelId);
    state.activePipeline = pipeline;
    state.activeModelId = modelId;
    state.activePipelineModelId = modelId;
    if (pipeline?.manifest?.modelType) {
      state.modelTypeCache[modelId] = normalizeModelType(pipeline.manifest.modelType);
    }
    state.modeModelId[modeKey] = modelId;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? null;
    updateMemoryControls();
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    return pipeline;
  } finally {
    await hideProgressOverlay();
  }
}

async function ensureRunPipeline() {
  const modelId = getSelectedModelId();
  const modeKey = state.uiMode === 'embedding'
    ? 'embedding'
    : (state.uiMode === 'translate' ? 'translate' : 'run');
  setRunLoading(true);
  try {
    return await ensurePipeline(modelId, 'Preparing Local Model', modeKey);
  } finally {
    setRunLoading(false);
  }
}

async function ensureDiffusionPipeline() {
  const modelId = getSelectedModelId();
  state.diffusionLoading = true;
  updateStatusIndicator();
  try {
    return await ensurePipeline(modelId, 'Preparing Local Model', 'diffusion');
  } finally {
    state.diffusionLoading = false;
    updateStatusIndicator();
  }
}

function drawDiffusionCanvas(result) {
  const canvas = $('diffusion-canvas');
  if (!canvas || !result) return;
  canvas.width = result.width;
  canvas.height = result.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const imageData = new ImageData(result.pixels, result.width, result.height);
  ctx.putImageData(imageData, 0, 0);
}

async function handleDiffusionRun() {
  if (state.diffusionGenerating || state.diffusionLoading) return;
  const promptEl = $('diffusion-prompt');
  const negativeEl = $('diffusion-negative');
  const stepsEl = $('diffusion-steps');
  const guidanceEl = $('diffusion-guidance');
  const seedEl = $('diffusion-seed');
  const widthEl = $('diffusion-width');
  const heightEl = $('diffusion-height');

  const request = {
    prompt: promptEl?.value?.trim() || '',
    negativePrompt: negativeEl?.value?.trim() || '',
    steps: stepsEl?.value ? Number(stepsEl.value) : undefined,
    guidanceScale: guidanceEl?.value ? Number(guidanceEl.value) : undefined,
    seed: seedEl?.value ? Number(seedEl.value) : undefined,
    width: widthEl?.value ? Number(widthEl.value) : undefined,
    height: heightEl?.value ? Number(heightEl.value) : undefined,
  };
  state.lastDiffusionRequest = { ...request };

  updateDiffusionStatus('Preparing...');
  state.diffusionGenerating = true;
  updateStatusIndicator();
  try {
    const pipeline = await ensureDiffusionPipeline();
    if (!pipeline.generate) {
      throw new Error('Selected model does not support diffusion generation.');
    }
    if (!pipeline.manifest || pipeline.manifest.modelType !== 'diffusion') {
      throw new Error('Selected model is not a diffusion model.');
    }
    updateDiffusionStatus('Generating...');
    const result = await pipeline.generate(request);
    if (result) {
      state.lastDiffusionRequest = {
        ...state.lastDiffusionRequest,
        width: result.width,
        height: result.height,
      };
    }
    if (!Number.isFinite(result?.width) || result.width <= 0 || !Number.isFinite(result?.height) || result.height <= 0) {
      throw new Error('Diffusion output dimensions are invalid.');
    }
    drawDiffusionCanvas(result);
    state.lastInferenceStats = pipeline.getStats?.() ?? null;
    state.lastMemoryStats = pipeline.getMemoryStats?.() ?? state.lastMemoryStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    updateDiffusionStatus('Complete');
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  } catch (error) {
    log.error('DopplerDemo', `Diffusion run failed: ${error.message}`);
    updateDiffusionStatus(`Error: ${error.message}`);
  } finally {
    state.diffusionGenerating = false;
    updateStatusIndicator();
  }
}

function handleDiffusionClear() {
  const canvas = $('diffusion-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
  updateDiffusionStatus('Idle');
}

async function preloadEnergyPipelineIfNeeded() {
  if (state.uiMode !== 'energy') return;
  if (state.energyLoading || state.energyGenerating) return;

  const modelId = getSelectedModelId();
  if (!modelId) return;

  const selectedModelType = normalizeModelType(await getModelTypeForId(modelId));
  if (selectedModelType !== 'energy') return;

  const activeModelType = normalizeModelType(state.activePipeline?.manifest?.modelType);
  if (
    state.activePipeline &&
    state.activeModelId === modelId &&
    activeModelType === 'energy'
  ) {
    return;
  }

  updateEnergyStatus('Loading energy model...');
  try {
    await ensureEnergyPipeline();
    if (!state.energyGenerating) updateEnergyStatus('Ready');
  } catch (error) {
    log.warn('DopplerDemo', `Energy preload skipped: ${error.message}`);
    if (!state.energyGenerating) updateEnergyStatus('Idle');
  }
}

async function ensureEnergyPipeline() {
  const modelId = getSelectedModelId();
  state.energyLoading = true;
  updateStatusIndicator();
  try {
    return await ensurePipeline(modelId, 'Preparing Local Model', 'energy');
  } finally {
    state.energyLoading = false;
    updateStatusIndicator();
  }
}

async function runStandaloneQuintelPipeline(request) {
  const { EnergyPipeline } = await import('../src/energy/index.js');
  const pipeline = new EnergyPipeline();
  await pipeline.initialize({
    runtimeConfig: getRuntimeConfig(),
  });
  pipeline.manifest = {
    modelId: 'quintel-standalone',
    modelType: 'energy',
    energy: {},
  };
  try {
    return await pipeline.generate(request);
  } finally {
    await pipeline.unload();
  }
}

async function handleEnergyRun() {
  if (state.energyGenerating || state.energyLoading) return;
  const demo = getEnergyDemoById(state.energyDemoId) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  const problem = 'quintel';
  const size = readOptionalNumber($('energy-quintel-size'), { integer: true });
  const displayThreshold = readOptionalNumber($('energy-quintel-threshold'));
  const countTarget = readOptionalNumber($('energy-quintel-count-target'), { integer: true });
  const mirrorX = $('energy-rule-mirror-x')?.checked ?? false;
  const mirrorY = $('energy-rule-mirror-y')?.checked ?? false;
  const diagonal = $('energy-rule-diagonal')?.checked ?? false;
  const countRule = $('energy-rule-count')?.checked ?? false;
  const symmetryWeight = readOptionalNumber($('energy-weight-symmetry'));
  const countWeight = readOptionalNumber($('energy-weight-count'));
  const binarizeWeight = readOptionalNumber($('energy-weight-binarize'));
  const initMode = $('energy-init-mode')?.value || undefined;
  const initSeed = readOptionalNumber($('energy-init-seed'), { integer: true });
  const initScale = readOptionalNumber($('energy-init-scale'));
  const steps = readOptionalNumber($('energy-steps'), { integer: true });
  const stepSize = readOptionalNumber($('energy-step-size'));
  const gradientScale = readOptionalNumber($('energy-gradient-scale'));
  const convergenceThreshold = readOptionalNumber($('energy-convergence'));

  const request = {
    problem,
    initMode,
    seed: initSeed,
    initScale,
    steps,
    stepSize,
    gradientScale,
    convergenceThreshold,
  };
  const quintelRules = {
    mirrorX,
    mirrorY,
    diagonal,
    count: countRule,
    center: false,
  };
  const quintel = {
    rules: quintelRules,
  };
  if (size != null) quintel.size = size;
  if (Number.isFinite(countTarget)) quintel.countTarget = countTarget;
  const weights = {};
  if (Number.isFinite(symmetryWeight)) weights.symmetry = symmetryWeight;
  if (Number.isFinite(countWeight)) weights.count = countWeight;
  if (Number.isFinite(binarizeWeight)) weights.binarize = binarizeWeight;
  if (Object.keys(weights).length) quintel.weights = weights;
  request.quintel = quintel;

  state.lastEnergyResult = null;
  state.lastEnergyRequest = {
    size,
    displayThreshold,
  };

  updateEnergyStatus('Preparing...');
  state.energyGenerating = true;
  updateStatusIndicator();
  try {
    let result = null;
    let pipelineForStats = null;
    const selectedModelId = getSelectedModelId();
    const selectedModelType = normalizeModelType(await getModelTypeForId(selectedModelId));
    const useStandaloneQuintel = selectedModelType !== 'energy';

    if (useStandaloneQuintel) {
      updateEnergyStatus('Running Quintel...');
      result = await runStandaloneQuintelPipeline(request);
    } else {
      pipelineForStats = await ensureEnergyPipeline();
      if (!pipelineForStats.generate) {
        throw new Error('Selected model does not support energy generation.');
      }
      if (!pipelineForStats.manifest || pipelineForStats.manifest.modelType !== 'energy') {
        throw new Error('Selected model is not an energy model.');
      }
      updateEnergyStatus('Running...');
      result = await pipelineForStats.generate(request);
    }
    state.lastEnergyResult = result;
    if (result?.shape) {
      state.lastEnergyRequest = {
        shape: result.shape,
        size: result.shape[0],
        displayThreshold,
      };
    }
    drawEnergyChart(result?.energyHistory || []);
    updateEnergyStats(result);
    renderEnergyBoard(result?.state, result?.shape ?? size, displayThreshold);
    state.lastInferenceStats = pipelineForStats?.getStats?.() ?? null;
    state.lastMemoryStats = pipelineForStats?.getMemoryStats?.() ?? state.lastMemoryStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`, 'energy');
    }
    updateEnergyStatus('Complete');
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
  } catch (error) {
    log.error('DopplerDemo', `Energy run failed: ${error.message}`);
    updateEnergyStatus(`Error: ${error.message}`);
  } finally {
    state.energyGenerating = false;
    updateStatusIndicator();
  }
}

function handleEnergyClear() {
  clearEnergyChart();
  clearEnergyBoard();
  updateEnergyStats(null);
  updateEnergyStatus('Idle');
  state.lastEnergyResult = null;
}

async function handleRunGenerate() {
  if (state.runGenerating || state.runLoading) return;
  if (isTranslateCompareEnabled()) {
    await handleTranslateCompareRun();
    return;
  }
  const promptEl = $('run-prompt');
  const outputEl = $('run-output');
  const prompt = promptEl?.value?.trim() || '';
  const isEmbeddingMode = state.uiMode === 'embedding';
  const isTranslateMode = state.uiMode === 'translate';
  const runResetKvToggle = $('run-reset-kv-toggle');
  const resetContextEachRun = !isEmbeddingMode && Boolean(runResetKvToggle?.checked);
  if (!prompt) {
    updateRunStatus(
      isEmbeddingMode
        ? 'Enter text to embed.'
        : (isTranslateMode ? 'Enter text to translate.' : 'Enter a prompt to generate.')
    );
    return;
  }

  const translateSelection = isTranslateMode ? getTranslateLanguageSelection() : null;
  const translateRequest = isTranslateMode
    ? createTranslateTextRequest(
      prompt,
      translateSelection.sourceCode,
      translateSelection.targetCode
    )
    : null;
  let generationInput = isTranslateMode ? translateRequest : prompt;

  updateRunStatus('Preparing...');
  let pipeline;
  let modelType = null;
  try {
    pipeline = await ensureRunPipeline();
    modelType = normalizeModelType(pipeline?.manifest?.modelType);
    if (isEmbeddingMode && modelType !== 'embedding') {
      throw new Error('Selected model is not an embedding model.');
    }
    if (!isEmbeddingMode && (modelType === 'diffusion' || modelType === 'energy' || modelType === 'embedding')) {
      throw new Error('Selected model is not a text model.');
    }
    if (isTranslateMode && translateSelection) {
      const chatTemplateType = pipeline.manifest?.inference?.chatTemplate?.type;
      if (chatTemplateType !== 'translategemma') {
        // General chat model: build a plain instruction prompt instead of the structured
        // translate request, which only translategemma knows how to interpret.
        const { sourceCode, targetCode } = translateSelection;
        generationInput = `Translate the following from ${sourceCode} to ${targetCode}. Output only the translation, no explanation.\n\n${prompt}`;
      }
    }
    if (resetContextEachRun) {
      pipeline.reset?.();
    }
  } catch (error) {
    updateRunStatus(`Error: ${error.message}`);
    return;
  }

  const controller = new AbortController();
  state.runAbortController = controller;
  state.runPrefilling = !isEmbeddingMode;
  setRunGenerating(true);
  updateRunStatus(isEmbeddingMode ? 'Embedding...' : (isTranslateMode ? 'Translating...' : 'Generating...'));
  if (outputEl) outputEl.textContent = '';

  const options = buildRunGenerateOptions();
  const isEmbeddingModel = modelType === 'embedding';
  let output = '';
  let tokenCount = 0;
  const start = performance.now();
  let firstTokenAt = null;

  try {
    if (isEmbeddingModel) {
      const embedStart = performance.now();
      pipeline.reset?.();
      const result = await pipeline.embed(prompt, options);
      const queryEmbeddingValues = result?.embedding ?? new Float32Array(0);
      const querySummary = summarizeEmbeddingVector(queryEmbeddingValues);
      if (!Number.isFinite(querySummary.dimension) || querySummary.dimension <= 0) {
        throw new Error('No embedding returned.');
      }
      if (querySummary.nonFiniteCount > 0) {
        throw new Error(`Embedding contains non-finite values (${querySummary.nonFiniteCount}/${querySummary.dimension}).`);
      }
      const embeddingDocuments = refreshEmbeddingDemoDocuments({ force: true });
      updateRunStatus('Embedding reference documents...');
      const scoredDocuments = [];
      for (const doc of embeddingDocuments) {
        pipeline.reset?.();
        const docResult = await pipeline.embed(doc.text, options);
        const docEmbeddingValues = docResult?.embedding ?? new Float32Array(0);
        const docSummary = summarizeEmbeddingVector(docEmbeddingValues);
        const score = cosineSimilarity(queryEmbeddingValues, docEmbeddingValues);
        scoredDocuments.push({
          id: doc.id,
          title: doc.title,
          text: doc.text,
          tokens: Number.isFinite(docResult?.tokens?.length) ? docResult.tokens.length : 0,
          dimension: docSummary.dimension,
          nonFinite: docSummary.nonFiniteCount,
          score: Number.isFinite(score) ? Number(score.toFixed(6)) : null,
        });
      }

      const ranked = scoredDocuments
        .slice()
        .sort((a, b) => (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY))
        .map((entry, index) => ({ rank: index + 1, ...entry }));
      const embeddingMs = Math.max(1, performance.now() - embedStart);

      output = JSON.stringify(
        {
          mode: 'embedding',
          query: prompt,
          dimension: querySummary.dimension,
          tokens: result?.tokens?.length ?? 0,
          embedding_preview: querySummary.preview,
          retrieval: {
            documents: scoredDocuments,
            ranked,
            top_match: ranked[0]
              ? { id: ranked[0].id, title: ranked[0].title, score: ranked[0].score }
              : null,
          },
        },
        null,
        2
      );
      state.lastMetrics = {
        ...(state.lastMetrics || {}),
        embeddingDim: querySummary.dimension,
        embeddingMs: Number(embeddingMs.toFixed(2)),
      };
      if (outputEl) outputEl.textContent = output;
      updateRunStatus('Complete');
    } else {
      for await (const token of pipeline.generate(generationInput, {
        ...options,
        signal: controller.signal,
        ...(isTranslateMode ? { useChatTemplate: true } : {}),
      })) {
        if (controller.signal.aborted) break;
        output += token;
        tokenCount += 1;
        const now = performance.now();
        if (!firstTokenAt) {
          firstTokenAt = now;
          if (state.runPrefilling) {
            state.runPrefilling = false;
            updateStatusIndicator();
          }
        }
        if (firstTokenAt) {
          const elapsedDecode = Math.max(1, now - firstTokenAt);
          const liveTokensPerSec = tokenCount / (elapsedDecode / 1000);
          state.lastMetrics = {
            ...(state.lastMetrics || {}),
            liveTokensPerSec,
          };
        }
        if (outputEl) outputEl.textContent = output;
      }
      updateRunStatus(controller.signal.aborted ? 'Stopped' : 'Complete');
    }
  } catch (error) {
    if (controller.signal.aborted) {
      updateRunStatus('Stopped');
    } else {
      updateRunStatus(`Error: ${error.message}`);
    }
  } finally {
    const elapsed = Math.max(1, performance.now() - start);
    const tokensPerSec = tokenCount > 0 ? Number(((tokenCount / elapsed) * 1000).toFixed(2)) : null;
    state.lastMetrics = {
      ...(state.lastMetrics || {}),
      tokensPerSec,
      liveTokensPerSec: null,
    };
    if (translateSelection) {
      state.lastMetrics.translateSource = translateSelection.sourceCode;
      state.lastMetrics.translateTarget = translateSelection.targetCode;
      state.lastMetrics.translateRequest = translateRequest;
    }
    state.lastMemoryStats = pipeline?.getMemoryStats?.() ?? state.lastMemoryStats;
    state.lastInferenceStats = pipeline?.getStats?.() ?? state.lastInferenceStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    setRunGenerating(false);
    state.runAbortController = null;
  }
}

function stopRunGeneration() {
  if (state.runAbortController) {
    state.runAbortController.abort();
  }
}

function handleRunClear() {
  const promptEl = $('run-prompt');
  const outputEl = $('run-output');
  if (promptEl) {
    promptEl.value = '';
    setStarterExampleInput(promptEl, false);
  }
  if (outputEl) outputEl.textContent = '';
  for (const laneId of getCompareLaneIds()) {
    clearCompareLaneResult(laneId);
    renderTranslateCompareLane(laneId);
  }
  updateRunStatus('Idle');
  syncDeepLinkFromUI();
}

function handleInferencePulseReset() {
  state.lastMetrics = null;
  state.lastInferenceStats = null;
  state.lastMemoryStats = null;
  state.lastDiffusionRequest = null;
  state.lastEnergyRequest = null;
  state.runLog = [];
  state.runCounter = 0;
  for (const laneId of getCompareLaneIds()) {
    clearCompareLaneResult(laneId);
  }

  const snapshot = captureMemorySnapshot();
  updatePerformancePanel(snapshot);
  updateMemoryPanel(snapshot);
  renderRunLog();
  syncTranslateCompareUI();
}

function summarizeEmbeddingVector(values) {
  const dimension = Number.isFinite(values?.length) ? values.length : 0;
  let nonFiniteCount = 0;
  for (let i = 0; i < dimension; i++) {
    if (!Number.isFinite(values[i])) nonFiniteCount++;
  }
  return {
    dimension,
    nonFiniteCount,
    preview: Array.from(values.slice(0, Math.min(16, dimension))).map((v) => Number(v.toFixed(6))),
  };
}

function cosineSimilarity(a, b) {
  if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b)) return null;
  if (a.length !== b.length || a.length <= 0) return null;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const av = Number(a[i]);
    const bv = Number(b[i]);
    if (!Number.isFinite(av) || !Number.isFinite(bv)) return null;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA <= 0 || normB <= 0) return null;
  return dot / Math.sqrt(normA * normB);
}

async function unloadActivePipeline() {
  if (!state.activePipeline) return;
  try {
    await state.activePipeline.unload?.();
  } catch (error) {
    log.warn('DopplerDemo', `Unload failed: ${error.message}`);
  }
  state.activePipeline = null;
  state.activePipelineModelId = null;
  state.lastMemoryStats = null;
  state.lastInferenceStats = null;
  updateMemoryControls();
  const snapshot = captureMemorySnapshot();
  updateMemoryPanel(snapshot);
  updatePerformancePanel(snapshot);
}

async function clearAllMemory() {
  await unloadAllCompareLaneRuntimes();
  await unloadActivePipeline();
  destroyBufferPool();
  const snapshot = captureMemorySnapshot();
  updateMemoryPanel(snapshot);
  updatePerformancePanel(snapshot);
}

function startTelemetryLoop() {
  if (state.uiIntervalId) return;

  let telemetryInFlight = false;
  const tick = async () => {
    if (telemetryInFlight) return;
    telemetryInFlight = true;
    try {
      const now = Date.now();
      if (now - state.lastStorageRefresh > 15000) {
        state.lastStorageRefresh = now;
        await updateStorageInfo();
      }
      const snapshot = captureMemorySnapshot();
      updateMemoryPanel(snapshot);
      updatePerformancePanel(snapshot);
    } catch (error) {
      log.warn('DopplerDemo', `Telemetry update failed: ${error.message}`);
    } finally {
      telemetryInFlight = false;
    }
  };

  state.uiIntervalId = setInterval(() => {
    void tick();
  }, 1000);
  void tick();
}

function populateModelPresets() {
  const presetSelect = $('convert-model-preset');
  if (!presetSelect) return;
  presetSelect.innerHTML = '';
  const autoOpt = document.createElement('option');
  autoOpt.value = '';
  autoOpt.textContent = 'auto';
  presetSelect.appendChild(autoOpt);
  for (const presetId of listPresets()) {
    const opt = document.createElement('option');
    opt.value = presetId;
    opt.textContent = presetId;
    presetSelect.appendChild(opt);
  }
}

function populateRuntimePresetSelect(select, entries, fallbackValue) {
  if (!select) return;
  const previous = select.value;
  select.innerHTML = '';
  for (const entry of entries) {
    const opt = document.createElement('option');
    opt.value = entry.id;
    opt.textContent = entry.label;
    select.appendChild(opt);
  }
  const target = previous || fallbackValue;
  if (target !== undefined && entries.some((entry) => entry.id === target)) {
    select.value = target;
    return;
  }
  if (entries.length > 0) {
    select.value = entries[0].id;
  }
}

function populateRuntimePresetSelects() {
  const baseSelect = $('runtime-preset');
  const overrideSelect = $('runtime-config-preset');
  const baseEntries = RUNTIME_PRESET_REGISTRY.filter((entry) => entry.base);
  const overrideEntries = RUNTIME_PRESET_REGISTRY.filter((entry) => entry.override);
  populateRuntimePresetSelect(baseSelect, baseEntries, DEFAULT_RUNTIME_PRESET);
  populateRuntimePresetSelect(overrideSelect, overrideEntries, '');
}

function buildConverterConfig() {
  const presetSelect = $('convert-model-preset');
  const presetId = presetSelect?.value?.trim() || null;
  const weightSelect = $('convert-weight-dtype');
  const weightOverride = weightSelect?.value?.trim().toLowerCase() || null;

  const config = createConverterConfig();
  if (presetId) {
    config.presets.model = presetId;
  }
  if (weightOverride) {
    config.quantization.weights = weightOverride;
  }
  return config;
}

async function runConversion(files, converterConfig, label, modelIdOverride) {
  if (!isConversionSupported()) {
    throw new Error('Browser conversion requires OPFS or IndexedDB.');
  }
  if (modelIdOverride != null) {
    assertValidModelId(modelIdOverride, 'Conversion modelId');
  }
  updateConvertStatus(`Preparing conversion${label ? ` (${label})` : ''}...`, 0);
  state.convertActive = true;
  updateStatusIndicator();
  try {
    const resultModelId = await convertModel(files, {
      modelId: modelIdOverride || undefined,
      converterConfig,
      onProgress: (update) => {
        if (!update) return;
        const percent = Number.isFinite(update.percent) ? update.percent : null;
        const message = update.message || 'Converting...';
        updateConvertStatus(label ? `${message} (${label})` : message, percent);
      },
    });
    updateConvertStatus(`Conversion complete: ${resultModelId}`, 100);
    await refreshModelList();
  } finally {
    state.convertActive = false;
    updateStatusIndicator();
  }
}

function restoreParsedManifest(previousManifest) {
  if (previousManifest) {
    setManifest(previousManifest);
    return;
  }
  clearManifest();
}

async function detectRdrrImport(files) {
  const manifestFile = findPickedFileByBaseName(files, 'manifest.json');
  if (!manifestFile) {
    return { kind: 'none' };
  }

  const manifestText = await manifestFile.text();
  const previousManifest = getManifest();
  let manifest;
  try {
    manifest = parseManifest(manifestText);
  } catch (error) {
    return {
      kind: 'invalid',
      reason: `Found manifest.json but it is not a valid RDRR manifest: ${error.message}`,
    };
  } finally {
    restoreParsedManifest(previousManifest);
  }

  const shardFiles = new Map();
  const missing = [];
  for (const shard of manifest.shards || []) {
    const shardFile = findPickedFileByPath(files, shard.filename);
    if (!shardFile) {
      missing.push(shard.filename || `shard_${shard.index}`);
      continue;
    }
    shardFiles.set(shard.index, shardFile);
  }

  if (missing.length > 0) {
    const preview = missing.slice(0, 3).join(', ');
    const suffix = missing.length > 3 ? ` (+${missing.length - 3} more)` : '';
    return {
      kind: 'invalid',
      reason: `Found RDRR manifest, but shard files are missing: ${preview}${suffix}`,
    };
  }

  let tensorsFile = null;
  if (manifest.tensorsFile) {
    tensorsFile = findPickedFileByPath(files, manifest.tensorsFile);
    if (!tensorsFile) {
      return {
        kind: 'invalid',
        reason: `Found RDRR manifest, but missing tensor map file: ${manifest.tensorsFile}`,
      };
    }
  }

  return {
    kind: 'rdrr',
    manifest,
    manifestText,
    manifestFile,
    shardFiles,
    tensorsFile,
  };
}

async function importRdrrFromFiles(files, detection, label) {
  if (!detection || detection.kind !== 'rdrr') {
    throw new Error('RDRR import requires a valid manifest and shard set.');
  }

  const previousManifest = getManifest();
  state.convertActive = true;
  updateStatusIndicator();
  try {
    const manifest = parseManifest(detection.manifestText);
    const modelId = assertValidModelId(manifest.modelId, 'RDRR manifest modelId');

    await openModelStore(modelId);

    const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
    const totalSteps = shards.length + (manifest.tensorsFile ? 1 : 0) + 2;
    let completed = 0;
    const step = (message) => {
      completed += 1;
      const percent = totalSteps > 0 ? (completed / totalSteps) * 100 : 100;
      updateConvertStatus(label ? `${message} (${label})` : message, percent);
    };

    await saveManifest(JSON.stringify(manifest, null, 2));
    step(`Saved manifest for ${modelId}`);

    if (manifest.tensorsFile) {
      const tensorsFile = detection.tensorsFile || findPickedFileByPath(files, manifest.tensorsFile);
      if (!tensorsFile) {
        throw new Error(`Missing ${manifest.tensorsFile} for RDRR import.`);
      }
      const tensorsText = await tensorsFile.text();
      await saveTensorsToStore(tensorsText);
      step(`Saved ${manifest.tensorsFile}`);
    }

    const tokenizerFilePath = manifest.tokenizer?.file || null;
    let tokenizerJsonFile = tokenizerFilePath ? findPickedFileByPath(files, tokenizerFilePath) : null;
    let tokenizerModelFile = null;
    if (tokenizerJsonFile && getPathBaseName(getPickedFilePath(tokenizerJsonFile)) === 'tokenizer.model') {
      tokenizerModelFile = tokenizerJsonFile;
      tokenizerJsonFile = null;
    }
    if (!tokenizerJsonFile) {
      tokenizerJsonFile = findPickedFileByBaseName(files, 'tokenizer.json');
    }
    if (!tokenizerModelFile) {
      tokenizerModelFile = findPickedFileByBaseName(files, 'tokenizer.model');
    }

    if (tokenizerJsonFile) {
      await saveTokenizer(await tokenizerJsonFile.text());
    }
    if (tokenizerModelFile) {
      await saveTokenizerModel(await tokenizerModelFile.arrayBuffer());
    }

    for (const filename of AUX_IMPORT_FILENAMES) {
      const auxFile = findPickedFileByBaseName(files, filename);
      if (!auxFile) continue;
      await saveAuxFile(filename, await auxFile.arrayBuffer());
    }

    for (let i = 0; i < shards.length; i++) {
      const shard = shards[i];
      const shardFile = detection.shardFiles.get(shard.index) || findPickedFileByPath(files, shard.filename);
      if (!shardFile) {
        throw new Error(`Missing shard file: ${shard.filename}`);
      }
      const data = new Uint8Array(await shardFile.arrayBuffer());
      if (Number.isFinite(shard.size) && data.byteLength !== shard.size) {
        throw new Error(
          `Shard size mismatch for ${shard.filename}: expected ${shard.size} bytes, got ${data.byteLength}`
        );
      }
      await writeShard(shard.index, data, { verify: true });
      step(`Imported shard ${i + 1}/${shards.length}`);
    }

    await registerDownloadedModel(modelId);
    delete state.modelTypeCache[modelId];
    updateConvertStatus(`RDRR import complete: ${modelId}`, 100);
    await refreshModelList();
  } finally {
    restoreParsedManifest(previousManifest);
    state.convertActive = false;
    updateStatusIndicator();
  }
}

async function regenerateManifest(modelId) {
  if (!modelId) {
    throw new Error('Select a model before regenerating the manifest.');
  }

  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage.');
  }

  const manifest = parseManifest(manifestText);
  let tensorMap = manifest.tensors ?? null;
  if (!tensorMap && manifest.tensorsFile) {
    const tensorsText = await loadTensorsFromStore();
    if (!tensorsText) {
      throw new Error('tensors.json not found in storage.');
    }
    tensorMap = JSON.parse(tensorsText);
  }
  if (!tensorMap) {
    throw new Error('Manifest is missing tensor locations.');
  }

  const tensorNames = Object.keys(tensorMap);
  for (const name of tensorNames) {
    const entry = tensorMap[name];
    if (entry) {
      entry.role = classifyTensorRole(name);
    }
  }

  let inference = manifest.inference;
  if (manifest.modelType === 'diffusion') {
    if (!inference) {
      inference = { ...DEFAULT_MANIFEST_INFERENCE, presetId: 'diffusion' };
    }
  } else {
    const rawConfig = manifest.config ?? {};
    const architectureHint = rawConfig.architectures?.[0] ?? rawConfig.model_type ?? '';
    const presetId = manifest.inference?.presetId || detectPreset(rawConfig, architectureHint);
    if (presetId === 'transformer') {
      const modelType = rawConfig.model_type ?? 'unknown';
      throw new Error(
        `Unknown model family: architecture="${architectureHint || 'unknown'}", model_type="${modelType}"`
      );
    }
    const preset = resolvePreset(presetId);
    const modelConfig = rawConfig?.text_config ?? rawConfig ?? {};
    const hiddenSize = modelConfig.hidden_size ?? modelConfig.n_embd ?? modelConfig.d_model ?? modelConfig.model_dim ?? null;
    const numHeads = modelConfig.num_attention_heads ?? modelConfig.n_head ?? modelConfig.num_heads ?? null;
    const derivedHeadDim = (Number.isFinite(hiddenSize) && Number.isFinite(numHeads) && numHeads > 0)
      ? hiddenSize / numHeads
      : null;
    const configHeadDim = Number.isFinite(rawConfig.head_dim) ? rawConfig.head_dim : null;
    const manifestHeadDim = (
      manifest.architecture
      && typeof manifest.architecture === 'object'
      && Number.isFinite(manifest.architecture.headDim)
    )
      ? manifest.architecture.headDim
      : null;
    const headDim = configHeadDim
      ?? manifestHeadDim
      ?? (Number.isFinite(derivedHeadDim) && Math.floor(derivedHeadDim) === derivedHeadDim ? derivedHeadDim : null);
    if (!headDim) {
      throw new Error('Missing headDim in manifest config (head_dim or hidden_size/num_attention_heads).');
    }
    inference = buildManifestInference(
      preset,
      rawConfig,
      headDim,
      manifest.quantizationInfo ?? null,
      tensorNames
    );
  }

  const embeddingOutput = inferEmbeddingOutputConfig(tensorMap);
  if (embeddingOutput && inference?.output) {
    inference = {
      ...inference,
      output: {
        ...inference.output,
        ...embeddingOutput,
      },
    };
  }

  const updatedManifest = {
    ...manifest,
    inference,
    tensors: tensorMap,
    tensorCount: tensorNames.length,
    metadata: {
      ...(manifest.metadata || {}),
      manifestRegeneratedAt: new Date().toISOString(),
    },
  };

  await saveManifest(JSON.stringify(updatedManifest, null, 2));
  if (manifest.tensorsFile) {
    await saveTensorsToStore(JSON.stringify(tensorMap, null, 2));
  }

  return updatedManifest;
}

async function handleRegenerateManifest() {
  if (state.convertActive) return;
  const modelId = getSelectedModelId();
  updateConvertStatus(`Regenerating manifest${modelId ? ` (${modelId})` : ''}...`, 0);
  state.convertActive = true;
  updateStatusIndicator();
  try {
    await regenerateManifest(modelId);
    if (modelId) {
      delete state.modelTypeCache[modelId];
    }
    updateConvertStatus(`Manifest regenerated: ${modelId}`, 100);
    await refreshModelList();
  } catch (error) {
    log.error('DopplerDemo', `Manifest regenerate failed: ${error.message}`);
    updateConvertStatus(`Manifest error: ${error.message}`, 0);
  } finally {
    state.convertActive = false;
    updateStatusIndicator();
  }
}

async function handleConvertFiles() {
  if (state.convertActive) return;
  updateConvertStatus('Select a model folder or files...', 0);
  let files = null;
  let pickedLabel = null;
  try {
    const pickedDirectory = await pickModelDirectory();
    files = pickedDirectory?.files || null;
    pickedLabel = pickedDirectory?.directoryName || null;
  } catch (error) {
    files = null;
  }

  if (!files || files.length === 0) {
    const pickedFiles = await pickModelFiles({ multiple: true });
    files = pickedFiles?.files || null;
  }

  if (!files || files.length === 0) {
    updateConvertStatus('No model files found in the selected folder.', 0);
    return;
  }

  const hasWeights = files.some((file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.safetensors') || name.endsWith('.gguf');
  });

  const rdrrDetection = await detectRdrrImport(files);
  if (rdrrDetection.kind === 'rdrr') {
    updateConvertStatus(
      `Detected pre-converted RDRR package${pickedLabel ? ` in ${pickedLabel}` : ''}. Importing...`,
      0
    );
    await importRdrrFromFiles(files, rdrrDetection, pickedLabel);
    return;
  }
  if (rdrrDetection.kind === 'invalid' && !hasWeights) {
    updateConvertStatus(rdrrDetection.reason, 0);
    return;
  }
  if (rdrrDetection.kind === 'invalid' && hasWeights) {
    log.warn('DopplerDemo', rdrrDetection.reason);
  }

  if (!hasWeights) {
    updateConvertStatus('Missing .safetensors or .gguf in the selected folder.', 0);
    return;
  }

  const modelIdOverride = await deriveModelIdFromFiles(files, pickedLabel);
  if (!modelIdOverride) {
    updateConvertStatus(
      'Missing valid modelId. Use 2-128 chars: letters/numbers plus dot, underscore, hyphen.',
      0
    );
    return;
  }

  updateConvertStatus(
    `Found ${files.length} files${pickedLabel ? ` in ${pickedLabel}` : ''}. Starting conversion...`,
    0
  );
  const converterConfig = buildConverterConfig();
  await runConversion(files, converterConfig, pickedLabel, modelIdOverride);
}

async function handleConvertUrls() {
  const urlInput = $('convert-url-input');
  if (!urlInput) return;
  const urls = urlInput.value
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (!urls.length) return;

  const directRdrrBaseUrl = urls.length === 1 ? resolveDirectRdrrBaseUrlFromInput(urls[0]) : '';
  if (directRdrrBaseUrl) {
    updateConvertStatus('Detected direct RDRR manifest URL. Importing prebuilt package...', 0);
    await importRdrrFromBaseUrl(directRdrrBaseUrl);
    updateConvertStatus('Imported prebuilt RDRR package from manifest URL.', 100);
    return;
  }

  if (!state.quickModelCatalogLoading && getQuickCatalogEntries().length === 0) {
    await loadQuickModelCatalog();
  }
  const registryEntry = findQuickCatalogEntryForRegistryInput(urls);
  if (registryEntry) {
    updateConvertStatus(
      `Found ${registryEntry.modelId} in Doppler registry. Importing prebuilt RDRR instead of converting...`,
      0
    );
    await importQuickModelEntry(registryEntry);
    updateConvertStatus(`Imported prebuilt RDRR package: ${registryEntry.modelId}`, 100);
    return;
  }

  updateConvertStatus('No prebuilt RDRR match found in registry. Starting conversion...', 0);
  const converterConfig = buildConverterConfig();
  const sources = await createRemoteModelSources(urls, { converterConfig });
  await runConversion(sources, converterConfig);
}
// --- Exports ---
export {
  buildRunGenerateOptions,
  drawDiffusionCanvas,
  handleDiffusionClear,
  handleEnergyClear,
  stopRunGeneration,
  handleRunClear,
  handleInferencePulseReset,
  summarizeEmbeddingVector,
  cosineSimilarity,
  startTelemetryLoop,
  populateModelPresets,
  populateRuntimePresetSelect,
  populateRuntimePresetSelects,
  buildConverterConfig,
  restoreParsedManifest,
};
