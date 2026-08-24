import { getDevice, setTrackSubmits } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, readBufferSlice, uploadData } from '../../../../memory/buffer-pool.js';
import {
  runMatmul,
  runRMSNorm,
  runGeLU,
  runResidualAdd,
  runScale,
  runSoftmax,
  runSoftEmbeddingSplitF16,
  runSoftEmbeddingLogitsF16,
} from '../../../../gpu/kernel-selector.js';
import { runDiffusionGemmaCanvasStats } from '../../../../gpu/kernels/diffusion-gemma-sampling.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { embed } from '../embed.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, applySoftcapping } from '../logits/index.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer, getWeightDtype, getWeightMetadata, getLayout } from '../../../../gpu/weight-buffer.js';
import {
  getLogitsConfig,
  getLogitsWeights,
} from './logits-config.js';
import {
  assertTokenIdsInRange,
  assertTokenIdInRange,
  resolveStepOptions,
  resolveGenerateOptions,
  resolvePrefillOptions,
  resolvePrefillEmbeddingOptions,
  resolveAdvanceEmbeddingMode,
  getFinalNormWeights,
  extractEmbeddingFromHidden,
  extractTokenEmbeddingsFromHidden,
} from '../generator-runtime.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { getQKNormOnesBuffer } from '../attention/types.js';
import { FINITENESS_RESET_WORDS, borrowLinearWeight, borrowNormWeight, canUseChunkedSoftEmbeddingLogits, getExperimentalIntentBundleModule, normalizeCanvasTokenIds, normalizeSelectedLogitTokenIds, normalizeSelfConditioningLogits, normalizeSelfConditioningLogitsState, normalizeSelfConditioningSoftEmbeddingState, recordPrefillRecorderStats, releaseBorrowedWeight, resolveDiffusionGemmaSoftEmbeddingChunkRows, resolvePrefillChunkSubmitMode, resolveSuppressedSamplingTokenIds, traceActivationHealth } from './text.js';

export async function _createDiffusionGemmaSelfConditioningEmbeddings(canvasIds, selfConditioningLogits, opts) {
    const config = this._state.modelConfig;
    const canvasLength = canvasIds.length;
    const hiddenSize = config.hiddenSize;
    const vocabSize = config.vocabSize;
    const elementCount = canvasLength * hiddenSize;
    const softEmbeddingState = normalizeSelfConditioningSoftEmbeddingState(
      selfConditioningLogits,
      canvasLength,
      hiddenSize
    );
    const logitsState = softEmbeddingState
      ? null
      : normalizeSelfConditioningLogitsState(selfConditioningLogits, canvasLength, vocabSize);
    const logits = (softEmbeddingState || logitsState)
      ? null
      : normalizeSelfConditioningLogits(selfConditioningLogits, canvasLength, vocabSize);
    const weights = this._state.weights.get('diffusion_gemma_self_conditioning');
    if (!weights || typeof weights !== 'object') {
      throw new Error(
        'DiffusionGemma self-conditioning weights were not loaded. ' +
        'Expected model.decoder.self_conditioning tensors in the manifest.'
      );
    }

    const embedBufferRaw = this._state.weights.get('embed');
    if (!embedBufferRaw) {
      throw new Error('DiffusionGemma self-conditioning requires loaded embed_tokens weights.');
    }
    const borrowed = {
      preNorm: null,
      gateProj: null,
      upProj: null,
      downProj: null,
      softEmbedding: null,
    };
    let softLogitsTensor = null;
    let softLogitsOwned = true;
    let softEmbeddingStateTensor = null;
    let softEmbeddingStateOwned = false;
    let softmaxTensor = null;
    let softEmbeddings = null;
    let scaledSoftEmbeddings = null;
    let baseEmbeddings = null;
    let normedSoft = null;
    let gate = null;
    let up = null;
    let activated = null;
    let down = null;
    let combined = null;
    let output = null;

    const releaseTensorOnce = (() => {
      const released = new Set();
      return (tensor) => {
        const buffer = tensor?.buffer ?? null;
        if (!buffer || released.has(buffer)) return;
        released.add(buffer);
        releaseBuffer(buffer);
      };
    })();

    try {
      borrowed.preNorm = borrowNormWeight(weights.preNorm, 'diffusion_gemma_self_conditioning.pre_norm');
      borrowed.gateProj = borrowLinearWeight(weights.gateProj, 'diffusion_gemma_self_conditioning.gate_proj');
      borrowed.upProj = borrowLinearWeight(weights.upProj, 'diffusion_gemma_self_conditioning.up_proj');
      borrowed.downProj = borrowLinearWeight(weights.downProj, 'diffusion_gemma_self_conditioning.down_proj');
      if (isSplitWeightBuffer(embedBufferRaw) && this._state.embeddingTranspose === true) {
        throw new Error(
          'DiffusionGemma self-conditioning split embeddings require row-major embedding storage.'
        );
      }
      borrowed.softEmbedding = isSplitWeightBuffer(embedBufferRaw)
        ? null
        : borrowLinearWeight(embedBufferRaw, 'diffusion_gemma_self_conditioning.embed_tokens');

      const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
      const embedDtype = isCpuWeightBuffer(embedBufferRaw)
        ? embedBufferRaw.dtype
        : getWeightDtype(embedBufferRaw);
      const embedMetadata = getWeightMetadata(embedBufferRaw);
      baseEmbeddings = await embed(canvasIds, embedBuffer, {
        hiddenSize,
        vocabSize,
        scaleEmbeddings: config.scaleEmbeddings,
        embeddingScale: config.embeddingScale, embeddingNormalization: config.embeddingNormalization,
        debug: opts.debug,
        recorder: null,
        transpose: this._state.embeddingTranspose,
        activationDtype: 'f32',
        embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embedDtype }),
        embeddingStorageEncoding: embedMetadata?.storageEncoding ?? null,
        executionPolicies: this._state.executionV1State?.policies ?? null,
        operatorDiagnostics: this._state.operatorDiagnostics,
      });

      if (softEmbeddingState) {
        softEmbeddingStateOwned = softEmbeddingState.releaseOnUse;
        softEmbeddings = createTensor(
          softEmbeddingState.buffer,
          softEmbeddingState.dtype,
          [canvasLength, hiddenSize],
          'diffusion_gemma_self_conditioning_soft_embedding'
        );
        softEmbeddingStateTensor = softEmbeddings;
        if (!softEmbeddingState.scaled) {
          scaledSoftEmbeddings = await runScale(softEmbeddings, Math.sqrt(hiddenSize), {
            count: elementCount,
          });
          if (softEmbeddingStateOwned) {
            releaseTensorOnce(softEmbeddings);
          }
          softEmbeddings = scaledSoftEmbeddings;
          softEmbeddingStateTensor = null;
          scaledSoftEmbeddings = null;
        }
      } else if (logitsState || logits) {
        const softmaxTemperature = logitsState?.temperature ?? 1.0;
        if (logitsState) {
          softLogitsOwned = logitsState.releaseOnUse;
          softLogitsTensor = createTensor(
            logitsState.logitsBuffer,
            logitsState.logitsDtype,
            [canvasLength, vocabSize],
            'diffusion_gemma_self_conditioning_logits'
          );
        } else {
          const logitsBuffer = acquireBuffer(logits.byteLength, undefined, 'diffusion_gemma_self_conditioning_logits');
          uploadData(logitsBuffer, logits);
          softLogitsTensor = createTensor(logitsBuffer, 'f32', [canvasLength, vocabSize], 'diffusion_gemma_self_conditioning_logits');
        }
        if (canUseChunkedSoftEmbeddingLogits(
          logitsState,
          borrowed.softEmbedding?.value,
          this._state.embeddingTranspose
        )) {
          softEmbeddings = await runSoftEmbeddingLogitsF16(
            softLogitsTensor,
            borrowed.softEmbedding.value,
            canvasLength,
            hiddenSize,
            vocabSize,
            {
              temperature: softmaxTemperature,
              chunkRows: resolveDiffusionGemmaSoftEmbeddingChunkRows(this._state.runtimeConfig),
            }
          );
        } else {
          softmaxTensor = await runSoftmax(softLogitsTensor, -1, {
            batchSize: canvasLength,
            size: vocabSize,
            temperature: softmaxTemperature,
          });
          softEmbeddings = isSplitWeightBuffer(embedBufferRaw)
            ? await runSoftEmbeddingSplitF16(
              softmaxTensor,
              embedBufferRaw,
              canvasLength,
              hiddenSize,
              vocabSize
            )
            : await runMatmul(
              softmaxTensor,
              borrowed.softEmbedding.value,
              canvasLength,
              hiddenSize,
              vocabSize,
              {
                transposeB: this._state.embeddingTranspose === true,
                role: 'diffusion_gemma_self_conditioning_embed',
                outputDtype: 'f32',
                executionPolicies: this._state.executionV1State?.policies ?? null,
              }
            );
        }
        scaledSoftEmbeddings = await runScale(softEmbeddings, Math.sqrt(hiddenSize), {
          count: elementCount,
        });
        if (softEmbeddings !== softEmbeddingStateTensor || softEmbeddingStateOwned) {
          releaseTensorOnce(softEmbeddings);
        }
        softEmbeddings = scaledSoftEmbeddings;
        scaledSoftEmbeddings = null;
      } else {
        const zeroBuffer = acquireBuffer(elementCount * Float32Array.BYTES_PER_ELEMENT, undefined, 'diffusion_gemma_self_conditioning_zero');
        uploadData(zeroBuffer, new Uint8Array(elementCount * Float32Array.BYTES_PER_ELEMENT));
        softEmbeddings = createTensor(zeroBuffer, 'f32', [canvasLength, hiddenSize], 'diffusion_gemma_self_conditioning_zero');
      }

      normedSoft = await runRMSNorm(softEmbeddings, borrowed.preNorm.value, config.rmsNormEps, {
        batchSize: canvasLength,
        hiddenSize,
        rmsNormWeightOffset: false,
      });
      const intermediateSize = config.intermediateSize;
      gate = await runMatmul(
        normedSoft,
        borrowed.gateProj.value,
        canvasLength,
        intermediateSize,
        hiddenSize,
        {
          transposeB: 'auto',
          role: 'diffusion_gemma_self_conditioning_gate',
          outputDtype: 'f32',
          executionPolicies: this._state.executionV1State?.policies ?? null,
        }
      );
      up = await runMatmul(
        normedSoft,
        borrowed.upProj.value,
        canvasLength,
        intermediateSize,
        hiddenSize,
        {
          transposeB: 'auto',
          role: 'diffusion_gemma_self_conditioning_up',
          outputDtype: 'f32',
          executionPolicies: this._state.executionV1State?.policies ?? null,
        }
      );
      activated = await runGeLU(up, {
        size: canvasLength * intermediateSize,
        gate,
      });
      down = await runMatmul(
        activated,
        borrowed.downProj.value,
        canvasLength,
        hiddenSize,
        intermediateSize,
        {
          transposeB: 'auto',
          role: 'diffusion_gemma_self_conditioning_down',
          outputDtype: 'f32',
          executionPolicies: this._state.executionV1State?.policies ?? null,
        }
      );
      combined = await runResidualAdd(baseEmbeddings, down, elementCount, {
        executionPolicies: this._state.executionV1State?.policies ?? null,
      });
      const postNormWeight = weights.postNorm
        ? borrowNormWeight(weights.postNorm, 'diffusion_gemma_self_conditioning.post_norm')
        : {
          value: getQKNormOnesBuffer(hiddenSize),
          owned: false,
        };
      borrowed.postNorm = postNormWeight;
      output = await runRMSNorm(combined, postNormWeight.value, config.rmsNormEps, {
        batchSize: canvasLength,
        hiddenSize,
        rmsNormWeightOffset: false,
      });
      return output;
    } catch (error) {
      releaseTensorOnce(output);
      throw error;
    } finally {
      releaseTensorOnce(combined);
      releaseTensorOnce(down);
      releaseTensorOnce(activated);
      releaseTensorOnce(up);
      releaseTensorOnce(gate);
      releaseTensorOnce(normedSoft);
      if (softEmbeddings !== softEmbeddingStateTensor || softEmbeddingStateOwned) {
        releaseTensorOnce(softEmbeddings);
      }
      releaseTensorOnce(softmaxTensor);
      if (softLogitsOwned) {
        releaseTensorOnce(softLogitsTensor);
      }
      releaseTensorOnce(baseEmbeddings);
      releaseBorrowedWeight(borrowed.postNorm);
      releaseBorrowedWeight(borrowed.downProj);
      releaseBorrowedWeight(borrowed.upProj);
      releaseBorrowedWeight(borrowed.gateProj);
      releaseBorrowedWeight(borrowed.preNorm);
      releaseBorrowedWeight(borrowed.softEmbedding);
    }
  }

export async function _createDiffusionGemmaSelfConditioningSoftEmbeddingState(logitsState, canvasLength, hiddenSize, vocabSize) {
    const embedBufferRaw = this._state.weights.get('embed');
    if (!canUseChunkedSoftEmbeddingLogits(
      logitsState,
      embedBufferRaw,
      this._state.embeddingTranspose
    )) {
      return null;
    }

    const borrowed = borrowLinearWeight(embedBufferRaw, 'diffusion_gemma_self_conditioning.embed_tokens');
    let softEmbedding = null;
    let scaledSoftEmbedding = null;
    try {
      const logitsTensor = createTensor(
        logitsState.logitsBuffer,
        logitsState.logitsDtype,
        [canvasLength, vocabSize],
        'diffusion_gemma_self_conditioning_logits'
      );
      softEmbedding = await runSoftEmbeddingLogitsF16(
        logitsTensor,
        borrowed.value,
        canvasLength,
        hiddenSize,
        vocabSize,
        {
          temperature: logitsState.temperature,
          chunkRows: resolveDiffusionGemmaSoftEmbeddingChunkRows(this._state.runtimeConfig),
        }
      );
      scaledSoftEmbedding = await runScale(softEmbedding, Math.sqrt(hiddenSize), {
        count: canvasLength * hiddenSize,
      });
      const returnedBuffer = scaledSoftEmbedding.buffer;
      return {
        kind: 'soft_embedding',
        buffer: returnedBuffer,
        dtype: scaledSoftEmbedding.dtype,
        canvasLength,
        hiddenSize,
        scaled: true,
        releaseOnUse: true,
        release() {
          releaseBuffer(returnedBuffer);
        },
      };
    } finally {
      if (softEmbedding?.buffer && softEmbedding.buffer !== scaledSoftEmbedding?.buffer) {
        releaseBuffer(softEmbedding.buffer);
      }
      releaseBorrowedWeight(borrowed);
    }
  }

export async function computeDiffusionGemmaCanvasLogits(args, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    const canvasIds = normalizeCanvasTokenIds(args?.canvas, 'computeDiffusionGemmaCanvasLogits');
    if (canvasIds.length === 0) {
      throw new Error('[DiffusionGemma] computeDiffusionGemmaCanvasLogits requires at least one canvas token.');
    }
    this._assertTokenIdsInRange(canvasIds, 'computeDiffusionGemmaCanvasLogits.canvas');
    const seqLenBefore = this._state.currentSeqLen;
    const opts = {
      ...resolvePrefillOptions(this._state, {
        ...options,
        useChatTemplate: false,
      }),
      _diffusionGemmaDecoder: true,
    };
    let selfConditioned = null;
    let currentHiddenBuffer = null;
    try {
      selfConditioned = await this._createDiffusionGemmaSelfConditioningEmbeddings(
        canvasIds,
        args?.selfConditioningLogits ?? null,
        opts
      );
      const prefillResult = await this._prefillToHidden(canvasIds, {
        ...opts,
        embeddingOverrides: {
          offset: 0,
          prefixLength: canvasIds.length,
          embeddings: selfConditioned.buffer,
        },
      });
      const {
        numTokens,
        currentRecorder,
        recordProfile,
        debugCheckBuffer,
      } = prefillResult;
      currentHiddenBuffer = prefillResult.currentHiddenBuffer;
      if (currentRecorder) {
        recordPrefillRecorderStats(this._state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      } else {
        const device = getDevice();
        if (device) {
          await device.queue.onSubmittedWorkDone();
        }
      }
      const logits = await computeLogits(
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this._state),
        getLogitsConfig(this._state),
        this._state.useGPU,
        this._state.debugFlags,
        undefined,
        debugCheckBuffer,
        this._state.runtimeConfig.shared.debug.probes,
        { lastPositionOnly: false },
        this._state.operatorDiagnostics
      );
      const expected = canvasIds.length * this._state.modelConfig.vocabSize;
      if (logits.length !== expected) {
        throw new Error(
          `[DiffusionGemma] canvas logits length mismatch: expected ${expected}, got ${logits.length}.`
        );
      }
      return logits;
    } finally {
      this._state.currentSeqLen = seqLenBefore;
      if (currentHiddenBuffer) {
        releaseBuffer(currentHiddenBuffer);
      }
      if (selfConditioned?.buffer) {
        releaseBuffer(selfConditioned.buffer);
      }
    }
  }

export async function computeDiffusionGemmaCanvasStep(args, options = {}) {
    if (!this._state.isLoaded) throw new Error('Model not loaded');
    if (this._state.isGenerating && options.__internalGenerate !== true) {
      throw new Error('Generation already in progress');
    }
    const canvasIds = normalizeCanvasTokenIds(args?.canvas, 'computeDiffusionGemmaCanvasStep');
    if (canvasIds.length === 0) {
      throw new Error('[DiffusionGemma] computeDiffusionGemmaCanvasStep requires at least one canvas token.');
    }
    this._assertTokenIdsInRange(canvasIds, 'computeDiffusionGemmaCanvasStep.canvas');
    const temperature = args?.temperature;
    if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0) {
      throw new Error('[DiffusionGemma] computeDiffusionGemmaCanvasStep requires a positive temperature.');
    }
    const seqLenBefore = this._state.currentSeqLen;
    const opts = {
      ...resolvePrefillOptions(this._state, {
        ...options,
        useChatTemplate: false,
      }),
      _diffusionGemmaDecoder: true,
    };
    let selfConditioned = null;
    let currentHiddenBuffer = null;
    let logitsBuffer = null;
    let statsBuffers = null;
    try {
      selfConditioned = await this._createDiffusionGemmaSelfConditioningEmbeddings(
        canvasIds,
        args?.selfConditioningLogits ?? null,
        opts
      );
      const prefillResult = await this._prefillToHidden(canvasIds, {
        ...opts,
        embeddingOverrides: {
          offset: 0,
          prefixLength: canvasIds.length,
          embeddings: selfConditioned.buffer,
        },
      });
      const {
        numTokens,
        currentRecorder,
        recordProfile,
        debugCheckBuffer,
      } = prefillResult;
      currentHiddenBuffer = prefillResult.currentHiddenBuffer;
      if (currentRecorder) {
        recordPrefillRecorderStats(this._state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);
      } else {
        const device = getDevice();
        if (device) {
          await device.queue.onSubmittedWorkDone();
        }
      }
      if (selfConditioned?.buffer) {
        releaseBuffer(selfConditioned.buffer);
        selfConditioned = null;
      }

      const logitsResult = await computeLogitsGPU(
        currentHiddenBuffer,
        numTokens,
        getLogitsWeights(this._state),
        getLogitsConfig(this._state),
        this._state.debugFlags,
        this._state.operatorDiagnostics
      );
      if (!logitsResult?.logitsBuffer) {
        throw new Error('[DiffusionGemma] GPU canvas step requires GPU logits.');
      }
      logitsBuffer = logitsResult.logitsBuffer;
      if (currentHiddenBuffer) {
        releaseBuffer(currentHiddenBuffer);
        currentHiddenBuffer = null;
      }
      if (logitsResult.logitsDtype !== 'f32') {
        throw new Error(
          `[DiffusionGemma] GPU canvas stats require f32 logits, got "${logitsResult.logitsDtype}".`
        );
      }
      if (logitsResult.vocabSize !== this._state.modelConfig.vocabSize) {
        throw new Error(
          `[DiffusionGemma] canvas logits vocab mismatch: expected ${this._state.modelConfig.vocabSize}, ` +
          `got ${logitsResult.vocabSize}.`
        );
      }

      statsBuffers = await runDiffusionGemmaCanvasStats(logitsBuffer, {
        canvasLength: canvasIds.length,
        vocabSize: logitsResult.vocabSize,
        temperature,
        padTokenId: this._state.modelConfig.diffusionGemma?.padTokenId ?? null,
        logitSoftcap: this._state.modelConfig.finalLogitSoftcapping ?? 0,
      });
      const [argmaxData, entropyData] = await Promise.all([
        readBuffer(statsBuffers.argmaxBuffer, canvasIds.length * Uint32Array.BYTES_PER_ELEMENT),
        readBuffer(statsBuffers.entropyBuffer, canvasIds.length * Float32Array.BYTES_PER_ELEMENT),
      ]);
      const argmaxCanvas = Int32Array.from(new Uint32Array(argmaxData));
      const entropies = new Float32Array(entropyData);
      const logitsState = {
        logitsBuffer,
        logitsDtype: logitsResult.logitsDtype,
        vocabSize: logitsResult.vocabSize,
        canvasLength: canvasIds.length,
        temperature,
        releaseOnUse: true,
      };
      const selfConditioningState =
        await this._createDiffusionGemmaSelfConditioningSoftEmbeddingState(
          logitsState,
          canvasIds.length,
          this._state.modelConfig.hiddenSize,
          logitsResult.vocabSize
        );
      if (selfConditioningState) {
        releaseBuffer(logitsBuffer);
        logitsBuffer = null;
      } else {
        const returnedLogitsBuffer = logitsBuffer;
        logitsBuffer = null;
        logitsState.release = () => {
          releaseBuffer(returnedLogitsBuffer);
        };
      }
      return {
        argmaxCanvas,
        entropies,
        selfConditioningLogits: selfConditioningState ?? logitsState,
      };
    } finally {
      this._state.currentSeqLen = seqLenBefore;
      if (statsBuffers?.argmaxBuffer) {
        releaseBuffer(statsBuffers.argmaxBuffer);
      }
      if (statsBuffers?.entropyBuffer) {
        releaseBuffer(statsBuffers.entropyBuffer);
      }
      if (logitsBuffer) {
        releaseBuffer(logitsBuffer);
      }
      if (currentHiddenBuffer) {
        releaseBuffer(currentHiddenBuffer);
      }
      if (selfConditioned?.buffer) {
        releaseBuffer(selfConditioned.buffer);
      }
    }
  }
