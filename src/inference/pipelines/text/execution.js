import { getDevice, initDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { getUniformCacheStats } from '../../../gpu/uniform-cache.js';
import { getBufferPool as getGlobalBufferPool, readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { configurePerfGuards } from '../../../gpu/perf-guards.js';
import { MoERouter } from '../../moe-router.js';
import { DecodeBufferManager } from '../../decode-buffers.js';
import { DecodeRing } from '../../decode-ring.js';
import { applyPipelineContexts, restorePipelineContexts } from '../context.js';
import { createInitializedPipeline } from '../factory.js';
import { PipelineState } from './state.js';
import { PipelineGenerator } from './generator.js';
import { parseModelConfig } from './config.js';
import {
  initRoPEFrequencies,
  createKVCache,
  loadWeights,
  initMoERouter,
  initSpeculativeDecoder,
  fuseQKVWeights,
  initEmulation,
  destroyEmulation,
} from './init.js';
import { formatChatMessages } from './chat-format.js';
import {
  runKernelWarmup,
  applyModelBatchingRuntimeDefaults,
  resolveKernelPathState,
  initTokenizerFromManifest,
  assertManifestComputeLaneBinding,
} from './model-load.js';
import { resolvePerLayerInputsSession } from './generator/session-context.js';
import { getKernelPathActivationDtype } from '../../../config/kernel-path-loader.js';
import { applyPipelineDebugConfig } from './debug-utils.js';
import { resolveLayerPipeline } from './layer-plan.js';
import { compileExecutionPlanState, resolveActiveExecutionPlan } from './execution-plan.js';
import { assertDtypeConsistency } from './dtype-contract.js';
import { applyExecutionV1RuntimeConfig, hasExecutionV1 } from './execution-v1.js';
import { getPlatform } from '../../../config/platforms/loader.js';
import {
  createLinearAttentionRuntime,
  hasLinearAttentionLayers,
  resetLinearAttentionRuntime,
  restoreLinearAttentionRuntime,
} from './linear-attention.js';
import { getDopplerLoader } from '../../../loader/doppler-loader.js';
import { registerPipeline, getPipelineFactory } from '../registry.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { createObservationContext } from '../../observation-context.js';
import { createResolvedRuntimeSession } from './resolved-runtime-session.js';
import { assertBundledAdapterAuthorized } from '../../../config/revocation-policy.js';
import { assertNotAborted } from './abort-contract.js';
import {
  assertMultimodalSequenceCapacity,
  buildConservativeMultimodalGenerationOptions,
  expandImagePlaceholderTokenIds,
  resolveMultimodalMaxTokens,
  resolveSingleSpecialTokenId,
} from './modality-token-contract.js';
import { initConvLayerState } from './ops.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runEmbeddingPool } from '../../../gpu/kernels/embedding-pool.js';
import { runProbes } from './probes.js';
import { destroyPleBufferCache, destroyPleRuntimeCache } from './per-layer-inputs.js';
import {
  initialize as initializeImpl,
  loadModel as loadModelImpl,
  _loadWeights as _loadWeightsImpl,
  _initRoPE as _initRoPEImpl,
  _initConvLayerStates as _initConvLayerStatesImpl,
  _loadVisionWeights as _loadVisionWeightsImpl,
  _ensureVisionWeightsLoaded as _ensureVisionWeightsLoadedImpl,
  _loadAudioWeights as _loadAudioWeightsImpl,
  _ensureAudioWeightsLoaded as _ensureAudioWeightsLoadedImpl,
} from './lifecycle.js';

async function readMeanPooledEmbedding(featuresBuffer, numTokens, hiddenSize, label) {
  const pooled = await runEmbeddingPool(
    createTensor(featuresBuffer, 'f32', [numTokens, hiddenSize], label),
    { rowCount: numTokens, hiddenSize, mode: 'mean' }
  );
  try {
    const bytes = await readBuffer(
      pooled.buffer,
      hiddenSize * Float32Array.BYTES_PER_ELEMENT
    );
    return new Float32Array(bytes);
  } finally {
    releaseBuffer(pooled.buffer);
  }
}

export function createImageTranscriptionResourceScope(
  pipeline,
  encodeResult,
  releaseFeature = releaseBuffer
) {
  if (!pipeline || typeof pipeline !== 'object') {
    throw new Error('[Pipeline] Image transcription resource scope requires a pipeline.');
  }
  if (!encodeResult || typeof encodeResult !== 'object') {
    throw new Error('[Pipeline] Image transcription resource scope requires an encode result.');
  }
  if (typeof releaseFeature !== 'function') {
    throw new Error('[Pipeline] Image transcription resource scope requires a release function.');
  }

  const originalRopeCos = pipeline.ropeFreqsCos;
  const originalRopeSin = pipeline.ropeFreqsSin;
  let glmOcrRopeOverride = null;
  let hasRun = false;
  let runActive = false;
  let hasFinished = false;
  let generationStarted = false;

  return {
    setGlmOcrRopeOverride(override) {
      if (generationStarted || hasFinished) {
        throw new Error('[Pipeline] Cannot register an MRoPE override after transcription started.');
      }
      if (
        !override
        || typeof override !== 'object'
        || !override.cos
        || !override.sin
        || typeof override.release !== 'function'
      ) {
        throw new Error('[Pipeline] GLM-OCR MRoPE override is missing cos, sin, or release.');
      }
      if (glmOcrRopeOverride) {
        throw new Error('[Pipeline] GLM-OCR MRoPE override is already registered.');
      }
      glmOcrRopeOverride = override;
      pipeline.ropeFreqsCos = override.cos;
      pipeline.ropeFreqsSin = override.sin;
    },

    async runGeneration(task) {
      if (!runActive) {
        throw new Error('[Pipeline] Image transcription generation requires an active resource scope.');
      }
      if (generationStarted) {
        throw new Error('[Pipeline] Image transcription generation may run only once.');
      }
      if (typeof task !== 'function') {
        throw new Error('[Pipeline] Image transcription generation requires a task.');
      }
      generationStarted = true;
      return task();
    },

    async run(task) {
      if (hasRun) {
        throw new Error('[Pipeline] Image transcription resource scope may run only once.');
      }
      if (typeof task !== 'function') {
        throw new Error('[Pipeline] Image transcription resource scope requires a task.');
      }
      hasRun = true;
      runActive = true;
      try {
        return await task();
      } finally {
        runActive = false;
        hasFinished = true;
        try {
          if (glmOcrRopeOverride) {
            pipeline.ropeFreqsCos = originalRopeCos;
            pipeline.ropeFreqsSin = originalRopeSin;
            glmOcrRopeOverride.release();
          }
        } finally {
          if (encodeResult.features) {
            releaseFeature(encodeResult.features);
          }
        }
      }
    },
  };
}

export async function transcribeImage({ imageBytes, width, height, prompt, maxTokens, softTokenBudget, signal }) {
    assertNotAborted(signal);
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support image transcription (no image_token_id in manifest).'
      );
    }
    await this._ensureVisionWeightsLoaded();
    assertNotAborted(signal);

    this.reset();

    // Lazy-load vision module (avoids GPU kernel dependency for text-only pipelines)
    const { encodeImage } = await import('../vision/index.js');

    // Step 1: Encode image through vision pipeline
    const probeVisionTensor = async (stage, buffer, options = {}) => runProbes(stage, buffer, {
      ...options,
      probes: this.runtimeConfig.shared.debug.probes,
      recorder: null,
      operatorDiagnostics: this.operatorDiagnostics,
      dtype: options.dtype ?? 'f32',
    });
    const encodeResult = await encodeImage({
      pixels: imageBytes,
      width,
      height,
      visionConfig: this.visionConfig,
      weights: this.visionWeights,
      softTokenBudget,
      probeTensor: probeVisionTensor,
    });
    await probeVisionTensor('image_features_out', encodeResult.features, {
      numTokens: encodeResult.numTokens,
      hiddenSize: this.visionConfig.outHiddenSize,
    });
    const transcriptionResources = createImageTranscriptionResourceScope(this, encodeResult);

    return transcriptionResources.run(async () => {
      // Step 2: Build the multimodal prompt from the model's chat template and
      // expand the single <|image|> placeholder into the exact visual-token span.
      const visionArchitecture = this.visionConfig?.visionArchitecture;
      const isGemma4 = visionArchitecture === 'gemma4';
      const isGlmOcr = visionArchitecture === 'glmocr';
      if (!isGemma4 && !isGlmOcr) {
        throw new Error(
          `[Pipeline] transcribeImage: unsupported vision architecture "${visionArchitecture ?? 'unknown'}". ` +
          'Supported transcription paths: Gemma 4 and GLM-OCR.'
        );
      }
      const requestedPrompt = prompt ?? (isGlmOcr
        ? 'Text Recognition:'
        : 'Describe the image in one short sentence.');
      const imageTokenId = this.visionConfig?.imageTokenId ?? this.modelConfig?.imageTokenId;
      if (imageTokenId == null) {
        throw new Error(
          'Pipeline missing image_token_id. Re-convert the model with image token metadata.'
        );
      }
      const expectedTemplateType = isGlmOcr ? 'glmocr' : 'gemma4';
      const templateType = this.modelConfig?.chatTemplateType ?? expectedTemplateType;
      if (templateType !== expectedTemplateType) {
        throw new Error(
          `[Pipeline] transcribeImage: ${visionArchitecture} requires chatTemplate.type="${expectedTemplateType}", ` +
          `got ${JSON.stringify(templateType)}.`
        );
      }
      const chatTemplateThinking = this.modelConfig?.chatTemplateThinking;
      const chatOptions = chatTemplateThinking == null
        ? undefined
        : { thinking: chatTemplateThinking };
      const multimodalPrompt = formatChatMessages([
        {
          role: 'user',
          content: [
            { type: 'image' },
            { type: 'text', text: requestedPrompt },
          ],
        },
      ], templateType, chatOptions);
      const promptTokenIds = this.tokenizer.encode(multimodalPrompt);
      const imageTokenSpanLength = encodeResult.numTokens;
      const effectiveBudget = softTokenBudget ?? this.visionConfig?.defaultOutputLength;
      const maxImageTokenSpanLength = Number(effectiveBudget);
      if (!Number.isFinite(maxImageTokenSpanLength) || maxImageTokenSpanLength < 1 || Math.floor(maxImageTokenSpanLength) !== maxImageTokenSpanLength) {
        throw new Error(
          `[Pipeline] transcribeImage: invalid soft token budget ${effectiveBudget}. ` +
          'Expected a positive integer from the resolved vision config or softTokenBudget parameter.'
        );
      }
      if (imageTokenSpanLength > maxImageTokenSpanLength) {
        throw new Error(
          `[Pipeline] transcribeImage: encoded ${visionArchitecture} image produced ${imageTokenSpanLength} soft tokens, ` +
          `which exceeds the effective soft token budget=${maxImageTokenSpanLength}.`
        );
      }
      const expansionOptions = isGemma4
        ? {
            boiTokenId: resolveSingleSpecialTokenId(this.tokenizer, '<|image>', 'Gemma 4 BOI token'),
            eoiTokenId: resolveSingleSpecialTokenId(this.tokenizer, '<image|>', 'Gemma 4 EOI token'),
          }
        : {};
      const { inputIds: fullTokenIds, imageStartOffset } = expandImagePlaceholderTokenIds(
        promptTokenIds,
        imageTokenId,
        imageTokenSpanLength,
        expansionOptions
      );
      const padTokenId = this.tokenizer?.getSpecialTokens?.()?.pad;
      if (!Number.isFinite(padTokenId) || Math.floor(padTokenId) !== padTokenId || padTokenId < 0) {
        throw new Error(
          `[Pipeline] transcribeImage: ${visionArchitecture} multimodal prefill requires a tokenizer pad token ID, got ${padTokenId}.`
        );
      }

      // Step 3: Generate with embedding override at the image token offset.
      const tokens = [];
      const maxGen = resolveMultimodalMaxTokens(this.runtimeConfig, maxTokens);
      assertMultimodalSequenceCapacity({
        inputTokenCount: fullTokenIds.length,
        maxTokens: maxGen,
        maxSeqLen: this.kvCache?.maxSeqLen,
      });
      const stopTokenIds = this.modelConfig.stopTokenIds;

      if (isGlmOcr) {
        if (
          this.modelConfig.ropeInterleaved !== true
          || this.modelConfig.mropeInterleaved !== true
          || !Array.isArray(this.modelConfig.mropeSection)
        ) {
          throw new Error(
            '[Pipeline] transcribeImage: GLM-OCR requires interleaved 3-axis MRoPE metadata.'
          );
        }
        const { buildGlmOcrRopePositionPlan, uploadGlmOcrRopeFrequencies } = await import(
          '../vision/glmocr-rope.js'
        );
        const [gridTemporal, gridHeight, gridWidth] = encodeResult.gridThw ?? [];
        if (gridTemporal !== 1) {
          throw new Error(
            `[Pipeline] transcribeImage: GLM-OCR image path requires grid temporal extent 1, got ${gridTemporal}.`
          );
        }
        const positionPlan = buildGlmOcrRopePositionPlan({
          promptLength: fullTokenIds.length,
          capacity: fullTokenIds.length + maxGen,
          imageStartOffset,
          imageTokenLength: imageTokenSpanLength,
          gridHeight,
          gridWidth,
          mergeSize: this.visionConfig.spatialMergeSize,
        });
        const glmOcrRopeOverride = await uploadGlmOcrRopeFrequencies(positionPlan, {
          rotaryDim: this.modelConfig.ropeRotaryDim,
          frequencyBaseDim: this.modelConfig.ropeFrequencyBaseDim,
          ropeTheta: this.modelConfig.ropeTheta,
          mropeSection: this.modelConfig.mropeSection,
        });
        transcriptionResources.setGlmOcrRopeOverride(glmOcrRopeOverride);
      }

      const generationOptions = {
        inputIds: fullTokenIds,
        embeddingOverrides: {
          prefixLength: encodeResult.numTokens,
          offset: imageStartOffset,
          embeddings: encodeResult.features,
        },
        __internalEmbeddingInputSpan: {
          offset: imageStartOffset,
          length: encodeResult.numTokens,
          tokenId: padTokenId,
        },
        ...(isGemma4
          ? {
              __internalMultimodalBidirectionalSpan: {
                offset: imageStartOffset,
                length: encodeResult.numTokens,
              },
            }
          : {}),
        maxTokens: maxGen,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      };
      const generation = await transcriptionResources.runGeneration(
        () => this.generator.generateTokenIds(
          '',
          buildConservativeMultimodalGenerationOptions(generationOptions)
        )
      );
      for (const token of generation.tokenIds ?? []) {
        if (Array.isArray(stopTokenIds) && stopTokenIds.includes(token)) break;
        tokens.push(token);
      }
      const text = this.tokenizer.decode(tokens);
      return { text, tokens };
    });
  }

export async function transcribeVideo({ frames, prompt, maxTokens, maxFrames, perFrameSoftTokenBudget, signal }) {
    assertNotAborted(signal);
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support video transcription (no image_token_id in manifest for vision encoder).'
      );
    }
    await this._ensureVisionWeightsLoaded();

    this.reset();

    // Lazy-load video module
    const { encodeVideo } = await import('../video/index.js');

    // Step 1: Encode video frames through vision pipeline
    const encodeResult = await encodeVideo({
      frames,
      visionConfig: this.visionConfig,
      weights: this.visionWeights,
      maxFrames: maxFrames ?? 8,
      perFrameSoftTokenBudget,
    });

    // Step 2: Build the multimodal prompt with <|video|> placeholder
    const requestedPrompt = prompt ?? 'Describe the video in one short sentence.';
    const videoTokenId = this.tokenizer?.model?.tokenToId?.('<|video_token|>')
      ?? this.tokenizer?.encode?.('<|video|>')?.find?.((id) => id !== undefined)
      ?? null;
    // Fall back to image token ID for video placeholder expansion
    const placeholderTokenId = videoTokenId ?? this.visionConfig?.imageTokenId ?? this.imageTokenId;
    if (placeholderTokenId == null) {
      throw new Error(
        'Pipeline missing video/image token ID for video placeholder expansion.'
      );
    }

    const templateType = this.modelConfig?.chatTemplateType ?? 'gemma4';
    const chatOptions = this.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined;
    const multimodalPrompt = formatChatMessages([
      {
        role: 'user',
        content: [
          { type: 'video' },
          { type: 'text', text: requestedPrompt },
        ],
      },
    ], templateType, chatOptions);
    const promptTokenIds = this.tokenizer.encode(multimodalPrompt);
    const videoTokenSpanLength = encodeResult.numTokens;

    // Resolve BOV/EOV tokens (reuse image BOI/EOI if video-specific ones don't exist)
    const bovTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<|video|>', 'Gemma 4 BOV token');
    const eovTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<video|>', 'Gemma 4 EOV token');

    const { inputIds: fullTokenIds, imageStartOffset: videoStartOffset } = expandImagePlaceholderTokenIds(
      promptTokenIds,
      placeholderTokenId,
      videoTokenSpanLength,
      { boiTokenId: bovTokenId, eoiTokenId: eovTokenId }
    );

    const padTokenId = this.tokenizer?.getSpecialTokens?.()?.pad;
    if (!Number.isFinite(padTokenId) || Math.floor(padTokenId) !== padTokenId || padTokenId < 0) {
      throw new Error(
        `[Pipeline] transcribeVideo: Gemma 4 multimodal prefill requires a tokenizer pad token ID, got ${padTokenId}.`
      );
    }

    // Step 3: Generate with embedding override at the video token offset
    const tokens = [];
    const maxGen = resolveMultimodalMaxTokens(this.runtimeConfig, maxTokens);
    const stopTokenIds = this.modelConfig.stopTokenIds;

    try {
      const generation = await this.generator.generateTokenIds('', buildConservativeMultimodalGenerationOptions({
        inputIds: fullTokenIds,
        embeddingOverrides: {
          prefixLength: encodeResult.numTokens,
          offset: videoStartOffset,
          embeddings: encodeResult.features,
        },
        __internalEmbeddingInputSpan: {
          offset: videoStartOffset,
          length: encodeResult.numTokens,
          tokenId: padTokenId,
        },
        __internalMultimodalBidirectionalSpan: {
          offset: videoStartOffset,
          length: encodeResult.numTokens,
        },
        maxTokens: maxGen,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      }));
      for (const token of generation.tokenIds ?? []) {
        if (Array.isArray(stopTokenIds) && stopTokenIds.includes(token)) break;
        tokens.push(token);
      }
    } finally {
      if (encodeResult.features) {
        releaseBuffer(encodeResult.features);
      }
    }

    const text = this.tokenizer.decode(tokens);
    return { text, tokens };
  }

export async function transcribeAudio({ audio, prompt, maxTokens, signal }) {
    assertNotAborted(signal);
    if (!this.audioCapable) {
      throw new Error(
        'Pipeline does not support audio transcription (no audio_token_id in manifest).'
      );
    }
    await this._ensureAudioWeightsLoaded();

    this.reset();

    // Lazy-load audio modules
    const { encodeAudio } = await import('../audio/index.js');

    let encodeResult;
    if (this.audioConfig.depth === 0) {
      encodeResult = await encodeAudio({
        rawAudio: audio,
        audioConfig: this.audioConfig,
        weights: this.audioWeights,
      });
    } else {
      const { extractLogMelSpectrogram } = await import('../audio/mel.js');
      const { features: melFeatures, numFrames, nMels } = extractLogMelSpectrogram(audio);
      encodeResult = await encodeAudio({
        melFeatures,
        numFrames,
        nMels,
        audioConfig: this.audioConfig,
        weights: this.audioWeights,
      });
    }

    // Step 3: Build the multimodal prompt with <|audio|> placeholder
    const requestedPrompt = prompt ?? 'Transcribe the audio.';
    const audioTokenId = this.audioConfig?.audioTokenId ?? this.audioTokenId;
    if (audioTokenId == null) {
      throw new Error(
        'Pipeline missing audio_token_id. Re-convert the model with audio token metadata.'
      );
    }
    const templateType = this.modelConfig?.chatTemplateType ?? 'gemma4';
    const chatOptions = this.modelConfig?.chatTemplateThinking === true ? { thinking: true } : undefined;
    const multimodalPrompt = formatChatMessages([
      {
        role: 'user',
        content: [
          { type: 'audio' },
          { type: 'text', text: requestedPrompt },
        ],
      },
    ], templateType, chatOptions);
    const promptTokenIds = this.tokenizer.encode(multimodalPrompt);
    const audioTokenSpanLength = encodeResult.numTokens;

    // Resolve BOA/EOA tokens
    const boaTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<|audio|>', 'Gemma 4 BOA token');
    const eoaTokenId = resolveSingleSpecialTokenId(this.tokenizer, '<audio|>', 'Gemma 4 EOA token');

    // Expand single audio placeholder token into the full audio token span
    const { inputIds: fullTokenIds, imageStartOffset: audioStartOffset } = expandImagePlaceholderTokenIds(
      promptTokenIds,
      audioTokenId,
      audioTokenSpanLength,
      { boiTokenId: boaTokenId, eoiTokenId: eoaTokenId }
    );

    const padTokenId = this.tokenizer?.getSpecialTokens?.()?.pad;
    if (!Number.isFinite(padTokenId) || Math.floor(padTokenId) !== padTokenId || padTokenId < 0) {
      throw new Error(
        `[Pipeline] transcribeAudio: Gemma 4 multimodal prefill requires a tokenizer pad token ID, got ${padTokenId}.`
      );
    }

    // Step 4: Generate with embedding override at the audio token offset
    const tokens = [];
    const maxGen = resolveMultimodalMaxTokens(this.runtimeConfig, maxTokens);
    const stopTokenIds = this.modelConfig.stopTokenIds;

    try {
      const generation = await this.generator.generateTokenIds('', buildConservativeMultimodalGenerationOptions({
        inputIds: fullTokenIds,
        embeddingOverrides: {
          prefixLength: encodeResult.numTokens,
          offset: audioStartOffset,
          embeddings: encodeResult.features,
        },
        __internalEmbeddingInputSpan: {
          offset: audioStartOffset,
          length: encodeResult.numTokens,
          tokenId: padTokenId,
        },
        __internalMultimodalBidirectionalSpan: {
          offset: audioStartOffset,
          length: encodeResult.numTokens,
        },
        maxTokens: maxGen,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      }));
      for (const token of generation.tokenIds ?? []) {
        if (Array.isArray(stopTokenIds) && stopTokenIds.includes(token)) break;
        tokens.push(token);
      }
    } finally {
      if (encodeResult.features) {
        releaseBuffer(encodeResult.features);
      }
    }

    const text = this.tokenizer.decode(tokens);
    return { text, tokens };
  }

export async function embed(prompt, options = {}) {
    assertNotAborted(options?.signal);
    this.resetForBatch();
    try {
      const result = await this.prefillWithEmbedding(prompt, {
        ...options,
        __skipStateSnapshot: true,
      });
      assertNotAborted(options?.signal);
      return {
        embedding: result.embedding,
        tokens: result.tokens,
        seqLen: result.seqLen,
        embeddingMode: result.embeddingMode,
        phase: result.phase ?? null,
      };
    } finally {
      this.resetForBatch();
    }
  }

export async function embedBatch(prompts, options = {}) {
    if (!Array.isArray(prompts)) {
      throw new Error('embedBatch expects an array of prompts');
    }
    assertNotAborted(options?.signal);
    const batchOptions = { ...options, __skipStateSnapshot: true };
    const outputs = [];
    for (const prompt of prompts) {
      // Check between every prompt so a superseded revision drops the rest.
      assertNotAborted(options?.signal);
      outputs.push(await this.embed(prompt, batchOptions));
    }
    return outputs;
  }

export async function encodeSequence(sequence, options = {}) {
    assertNotAborted(options?.signal);
    const contract = this.manifest?.inference?.sequence ?? null;
    if (this.manifest?.inference?.supportsSequence !== true || !contract) {
      throw new Error('Model manifest does not declare sequence encoding support.');
    }
    if (typeof sequence !== 'string' || sequence.length === 0) {
      throw new Error('encodeSequence expects a non-empty sequence string.');
    }
    const includeTokenEmbeddings = options.includeTokenEmbeddings
      ?? contract.tokenEmbeddings;
    const includeLogits = options.includeLogits === true;
    if (includeTokenEmbeddings && contract.tokenEmbeddings !== true) {
      throw new Error('Model manifest does not permit token-level sequence embeddings.');
    }
    if (includeLogits && contract.logits !== true) {
      throw new Error('Model manifest does not permit sequence logits.');
    }
    const needsTokenEmbeddings = includeTokenEmbeddings || contract.pooledEmbedding !== null;

    this.resetForBatch();
    try {
      const result = await this.prefillWithEmbedding(sequence, {
        ...options,
        embeddingMode: contract.pooledEmbedding?.mode ?? 'last',
        __skipStateSnapshot: true,
        __returnTokenEmbeddings: needsTokenEmbeddings,
        __returnSequenceLogits: includeLogits,
        __sequencePooling: contract.pooledEmbedding,
      });
      assertNotAborted(options?.signal);
      const hiddenSize = this.modelConfig.hiddenSize;
      return {
        alphabet: contract.alphabet,
        tokens: result.tokens,
        tokenMask: result.tokenMask ?? new Uint8Array(result.tokens.length),
        includedTokenCount: result.includedTokenCount ?? 0,
        tokenEmbeddings: includeTokenEmbeddings ? result.tokenEmbeddings : null,
        pooledEmbedding: result.pooledSequenceEmbedding ?? null,
        logits: includeLogits ? result.logits : null,
        embeddingDim: hiddenSize,
        vocabSize: this.modelConfig.vocabSize,
        phase: result.phase ?? null,
      };
    } finally {
      this.resetForBatch();
    }
  }

export async function embedImage({ pixels, width, height, softTokenBudget, signal } = {}) {
    assertNotAborted(signal);
    if (!this.visionCapable) {
      throw new Error(
        'Pipeline does not support image embedding (no image_token_id in manifest).'
      );
    }
    if (pixels == null) {
      throw new Error('[Pipeline] embedImage: pixels are required.');
    }
    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
      throw new Error('[Pipeline] embedImage: width and height must be positive integers.');
    }
    await this._ensureVisionWeightsLoaded();
    this.reset();

    const { encodeImage } = await import('../vision/index.js');
    const encodeResult = await encodeImage({
      pixels,
      width,
      height,
      visionConfig: this.visionConfig,
      weights: this.visionWeights,
      softTokenBudget,
    });

    const hiddenSize = this.modelConfig.hiddenSize;
    const numTokens = encodeResult.numTokens;
    if (!Number.isFinite(numTokens) || numTokens < 1) {
      releaseBuffer(encodeResult.features);
      throw new Error(`[Pipeline] embedImage: encoder produced ${numTokens} soft tokens; expected >= 1.`);
    }
    try {
      const pooled = await readMeanPooledEmbedding(
        encodeResult.features,
        numTokens,
        hiddenSize,
        'image_embedding_features'
      );
      return {
        embedding: pooled,
        embeddingDim: hiddenSize,
        numTokens,
        embeddingMode: 'mean',
      };
    } finally {
      releaseBuffer(encodeResult.features);
    }
  }

export async function embedAudio({ audio, signal } = {}) {
    assertNotAborted(signal);
    if (!this.audioCapable) {
      throw new Error(
        'Pipeline does not support audio embedding (no audio_token_id in manifest).'
      );
    }
    if (audio == null) {
      throw new Error('[Pipeline] embedAudio: audio is required.');
    }
    await this._ensureAudioWeightsLoaded();
    this.reset();

    const { encodeAudio } = await import('../audio/index.js');

    let encodeResult;
    if (this.audioConfig.depth === 0) {
      encodeResult = await encodeAudio({
        rawAudio: audio,
        audioConfig: this.audioConfig,
        weights: this.audioWeights,
      });
    } else {
      const { extractLogMelSpectrogram } = await import('../audio/mel.js');
      const { features: melFeatures, numFrames, nMels } = extractLogMelSpectrogram(audio);
      encodeResult = await encodeAudio({
        melFeatures,
        numFrames,
        nMels,
        audioConfig: this.audioConfig,
        weights: this.audioWeights,
      });
    }

    const hiddenSize = Number(this.audioConfig?.outputProjDims ?? this.modelConfig?.hiddenSize);
    const numTokens = encodeResult.numTokens;
    if (!Number.isFinite(hiddenSize) || hiddenSize < 1) {
      releaseBuffer(encodeResult.features);
      throw new Error('[Pipeline] embedAudio: audioConfig.outputProjDims is missing or invalid.');
    }
    if (!Number.isFinite(numTokens) || numTokens < 1) {
      releaseBuffer(encodeResult.features);
      throw new Error(`[Pipeline] embedAudio: encoder produced ${numTokens} tokens; expected >= 1.`);
    }
    try {
      const pooled = await readMeanPooledEmbedding(
        encodeResult.features,
        numTokens,
        hiddenSize,
        'audio_embedding_features'
      );
      return {
        embedding: pooled,
        embeddingDim: hiddenSize,
        numTokens,
        embeddingMode: 'mean',
      };
    } finally {
      releaseBuffer(encodeResult.features);
    }
  }
