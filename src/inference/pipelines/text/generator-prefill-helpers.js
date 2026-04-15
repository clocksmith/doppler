
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { castF32ToF16 } from '../../../gpu/kernels/cast.js';
import { f32ToF16Array } from '../../kv-cache/types.js';
import { isGpuBufferInstance } from '../../../gpu/weight-buffer.js';
import { createTensor } from '../../../gpu/tensor.js';
import { assertImplicitDtypeTransitionAllowed } from './dtype-contract.js';
import { applyChatTemplate } from './init.js';
import { formatChatMessages } from './chat-format.js';

function isStructuredChatRequest(prompt) {
  return prompt != null
    && typeof prompt === 'object'
    && !Array.isArray(prompt)
    && Array.isArray(prompt.messages);
}

export function resolvePromptInput(state, prompt, useChatTemplate, contextLabel) {
  const chatOptions = state.modelConfig.chatTemplateThinking === true ? { thinking: true } : undefined;
  if (typeof prompt === 'string') {
    if (useChatTemplate && state.modelConfig.chatTemplateType) {
      if (state.modelConfig.chatTemplateType === 'translategemma') {
        throw new Error(
          `[Pipeline] ${contextLabel}: translategemma chat template requires structured messages. ` +
          'Pass { messages: [...] } instead of a plain string prompt.'
        );
      }
      return applyChatTemplate(prompt, state.modelConfig.chatTemplateType, chatOptions);
    }
    return prompt;
  }

  if (prompt != null && typeof prompt === 'object' && !Array.isArray(prompt) && 'messages' in prompt && !Array.isArray(prompt.messages)) {
    throw new Error(
      `[Pipeline] ${contextLabel}: prompt.messages must be an array of chat messages, got ${typeof prompt.messages}. ` +
      'Pass { messages: [{ role: "user", content: "..." }, ...] }.'
    );
  }
  const messages = isStructuredChatRequest(prompt)
    ? prompt.messages
    : (Array.isArray(prompt) ? prompt : null);
  if (!messages) {
    throw new Error(
      `[Pipeline] ${contextLabel}: prompt must be a string, chat message array, or { messages: [...] }.`
    );
  }
  const templateType = useChatTemplate ? state.modelConfig.chatTemplateType : null;
  return formatChatMessages(messages, templateType, chatOptions);
}

export function releasePerLayerInputBuffer(buffer, recorder, decodeBuffers, pleCache = null) {
  if (!buffer) {
    return;
  }
  const ownsBuffer = decodeBuffers?.ownsBuffer(buffer) ?? false;
  if (ownsBuffer) {
    return;
  }
  const cachedPleBuffer = pleCache?.ownedBuffers instanceof Set && pleCache.ownedBuffers.has(buffer);
  if (cachedPleBuffer) {
    return;
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
    return;
  }
  releaseBuffer(buffer);
}

export function normalizePrefixEmbeddingOverride(override, hiddenSize, numTokens, contextLabel) {
  if (override == null) {
    return null;
  }
  if (typeof override !== 'object') {
    throw new Error(`[Pipeline] ${contextLabel}: embeddingOverrides must be an object when provided.`);
  }

  const prefixLength = Number(override.prefixLength);
  if (!Number.isFinite(prefixLength) || prefixLength < 0 || Math.floor(prefixLength) !== prefixLength) {
    throw new Error(
      `[Pipeline] ${contextLabel}: embeddingOverrides.prefixLength must be a non-negative integer.`
    );
  }
  if (prefixLength === 0) {
    return null;
  }

  const offset = Number(override.offset ?? 0);
  if (!Number.isFinite(offset) || offset < 0 || Math.floor(offset) !== offset) {
    throw new Error(
      `[Pipeline] ${contextLabel}: embeddingOverrides.offset must be a non-negative integer.`
    );
  }
  if (offset + prefixLength > numTokens) {
    throw new Error(
      `[Pipeline] ${contextLabel}: embedding override offset=${offset} + prefixLength=${prefixLength} exceeds numTokens=${numTokens}.`
    );
  }

  const embeddings = override.embeddings ?? null;
  if (!embeddings) {
    throw new Error(`[Pipeline] ${contextLabel}: embeddingOverrides.embeddings is required when prefixLength > 0.`);
  }

  const expectedLength = prefixLength * hiddenSize;
  if (embeddings instanceof Float32Array) {
    if (embeddings.length !== expectedLength) {
      throw new Error(
        `[Pipeline] ${contextLabel}: embedding override length mismatch ` +
        `(expected=${expectedLength}, got=${embeddings.length}).`
      );
    }
  } else if (isGpuBufferInstance(embeddings)) {
    const expectedBytes = expectedLength * Float32Array.BYTES_PER_ELEMENT;
    if (embeddings.size < expectedBytes) {
      throw new Error(
        `[Pipeline] ${contextLabel}: embedding override GPUBuffer too small ` +
        `(expected>=${expectedBytes} bytes, got=${embeddings.size}).`
      );
    }
  } else {
    throw new Error(
      `[Pipeline] ${contextLabel}: embeddingOverrides.embeddings must be a Float32Array or GPUBuffer.`
    );
  }

  return {
    prefixLength,
    offset,
    embeddings,
    expectedLength,
    byteLength: expectedLength * Float32Array.BYTES_PER_ELEMENT,
    byteOffset: offset * hiddenSize * Float32Array.BYTES_PER_ELEMENT,
  };
}

export function resolvePrefillEmbeddingInputIds(inputIds, embeddingInputSpan, contextLabel) {
  if (embeddingInputSpan == null) {
    return inputIds;
  }
  if (typeof embeddingInputSpan !== 'object') {
    throw new Error(`[Pipeline] ${contextLabel}: embeddingInputSpan must be an object when provided.`);
  }

  const offset = Number(embeddingInputSpan.offset);
  const length = Number(embeddingInputSpan.length);
  const tokenId = Number(embeddingInputSpan.tokenId);
  if (!Number.isFinite(offset) || Math.floor(offset) !== offset || offset < 0) {
    throw new Error(`[Pipeline] ${contextLabel}: embeddingInputSpan.offset must be a non-negative integer.`);
  }
  if (!Number.isFinite(length) || Math.floor(length) !== length || length < 0) {
    throw new Error(`[Pipeline] ${contextLabel}: embeddingInputSpan.length must be a non-negative integer.`);
  }
  if (!Number.isFinite(tokenId) || Math.floor(tokenId) !== tokenId || tokenId < 0) {
    throw new Error(`[Pipeline] ${contextLabel}: embeddingInputSpan.tokenId must be a non-negative integer.`);
  }
  if (offset + length > inputIds.length) {
    throw new Error(
      `[Pipeline] ${contextLabel}: embeddingInputSpan offset=${offset} + length=${length} ` +
      `exceeds inputIds length ${inputIds.length}.`
    );
  }
  const replacedInputIds = Array.from(inputIds);
  replacedInputIds.fill(tokenId, offset, offset + length);
  return replacedInputIds;
}

export function resolvePrefillMultimodalBidirectionalSpan(inputIds, bidirectionalSpan, contextLabel) {
  if (bidirectionalSpan == null) {
    return null;
  }
  if (typeof bidirectionalSpan !== 'object') {
    throw new Error(`[Pipeline] ${contextLabel}: multimodalBidirectionalSpan must be an object when provided.`);
  }

  const offset = Number(bidirectionalSpan.offset);
  const length = Number(bidirectionalSpan.length);
  if (!Number.isFinite(offset) || Math.floor(offset) !== offset || offset < 0) {
    throw new Error(`[Pipeline] ${contextLabel}: multimodalBidirectionalSpan.offset must be a non-negative integer.`);
  }
  if (!Number.isFinite(length) || Math.floor(length) !== length || length < 1) {
    throw new Error(`[Pipeline] ${contextLabel}: multimodalBidirectionalSpan.length must be a positive integer.`);
  }
  if ((offset + length) > inputIds.length) {
    throw new Error(
      `[Pipeline] ${contextLabel}: multimodalBidirectionalSpan offset=${offset} + length=${length} ` +
      `exceeds inputIds length ${inputIds.length}.`
    );
  }
  return { offset, length };
}

export async function applyPrefixEmbeddingOverride(baseTensor, override, hiddenSize, contextLabel, executionPolicies = null) {
  if (!override) {
    return baseTensor;
  }
  if (baseTensor.dtype !== 'f32' && baseTensor.dtype !== 'f16') {
    throw new Error(
      `[Pipeline] ${contextLabel}: embedding overrides require f32 or f16 activations, got ${baseTensor.dtype}.`
    );
  }

  const device = getDevice();
  if (!device) {
    throw new Error(`[Pipeline] ${contextLabel}: GPU device is required for embedding overrides.`);
  }

  const outputBuffer = acquireBuffer(baseTensor.buffer.size, undefined, 'prefill_embedding_override');
  const targetElementOffset = override.offset * hiddenSize;
  const targetByteOffset = targetElementOffset * (baseTensor.dtype === 'f16' ? 2 : 4);
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(baseTensor.buffer, 0, outputBuffer, 0, baseTensor.buffer.size);
    if (baseTensor.dtype === 'f32') {
      if (isGpuBufferInstance(override.embeddings)) {
        encoder.copyBufferToBuffer(override.embeddings, 0, outputBuffer, override.byteOffset, override.byteLength);
        device.queue.submit([encoder.finish()]);
      } else {
        device.queue.submit([encoder.finish()]);
        uploadData(outputBuffer, override.embeddings, override.byteOffset);
      }
      return createTensor(outputBuffer, baseTensor.dtype, [...baseTensor.shape], 'prefill_embedding_override');
    }

    if (isGpuBufferInstance(override.embeddings)) {
      const overrideTensor = createTensor(
        override.embeddings,
        'f32',
        [override.prefixLength, hiddenSize],
        'prefill_embedding_override_f32'
      );
      assertImplicitDtypeTransitionAllowed({
        executionPolicies,
        fromDtype: 'f32',
        toDtype: 'f16',
        op: 'prefill_embedding_override',
        detail: 'Prefix embedding override would pack GPU-provided f32 features into an f16 activation buffer.',
      });
      const castedOverride = await castF32ToF16(overrideTensor);
      try {
        encoder.copyBufferToBuffer(
          castedOverride.buffer,
          0,
          outputBuffer,
          targetByteOffset,
          castedOverride.buffer.size
        );
        device.queue.submit([encoder.finish()]);
      } finally {
        releaseBuffer(castedOverride.buffer);
      }
    } else {
      assertImplicitDtypeTransitionAllowed({
        executionPolicies,
        fromDtype: 'f32',
        toDtype: 'f16',
        op: 'prefill_embedding_override',
        detail: 'Prefix embedding override would pack CPU-provided f32 features into an f16 activation buffer.',
      });
      const packedOverride = f32ToF16Array(override.embeddings);
      device.queue.submit([encoder.finish()]);
      uploadData(outputBuffer, packedOverride, targetByteOffset);
    }
    return createTensor(outputBuffer, baseTensor.dtype, [...baseTensor.shape], 'prefill_embedding_override');
  } catch (error) {
    releaseBuffer(outputBuffer);
    throw error;
  }
}

export function shouldDisablePrefillCommandBatching(state, opts, multimodalBidirectionalSpan) {
  if (
    opts?.disableCommandBatching === true
    || opts?.debug === true
    || (Array.isArray(opts?.debugLayers) && opts.debugLayers.length > 0)
  ) {
    return true;
  }
  if (state?.kvCache?.layout === 'bdpa_paged') {
    return true;
  }
  if (multimodalBidirectionalSpan == null) {
    return false;
  }
  if (state?.kvCache?.hasGPUCache?.() !== true) {
    return false;
  }
  // WORKAROUND: recorded prefill with live KV-cache writes regresses Gemma 4
  // multimodal logits on the first generated token. Keep this lane fail-closed
  // until the recorder/cache interaction is numerically matched to the direct path.
  return true;
}
