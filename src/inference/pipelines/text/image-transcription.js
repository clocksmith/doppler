import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { formatChatMessages } from './chat-format.js';
import { assertNotAborted } from './abort-contract.js';
import {
  assertMultimodalSequenceCapacity,
  buildConservativeMultimodalGenerationOptions,
  expandImagePlaceholderTokenIds,
  resolveMultimodalMaxTokens,
  resolveSingleSpecialTokenId,
} from './modality-token-contract.js';
import { runProbes } from './probes.js';

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

