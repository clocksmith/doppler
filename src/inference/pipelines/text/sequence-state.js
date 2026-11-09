import { getDevice } from '../../../gpu/device.js';
import { createKVCache } from './init.js';
import { createLinearAttentionRuntime, resetLinearAttentionRuntime, resetLinearAttentionSequence } from './linear-attention.js';
import { isConvLayerType } from './layer-input-execution.js';
import { KVCache } from '../../kv-cache/base.js';
import { cloneLinearAttentionRuntimeOnGPU } from './linear-attention/state-clone.js';
import { resolvePromptInput } from './generator/prompt-input.js';

// Handles are local to one loaded pipeline. Applications own naming, admission,
// retention and eviction; this module owns complete decoder memory and settlement.
/** @type {WeakMap<import('./sequence-state.js').GenerationContext, import('./sequence-state.js').ContextEntry>} */
const records = new WeakMap();
/** @type {WeakMap<import('./state.js').PipelineState, Set<import('./sequence-state.js').GenerationContext>>} */
const owners = new WeakMap();
/** @type {WeakMap<import('./state.js').PipelineState, import('./sequence-state.js').ContextLease>} */
const leases = new WeakMap();

/** @type {import('./sequence-state.js').supportsGenerationContext} */
export function supportsGenerationContext(state) {
  return state.useGPU === true && state.kvCache instanceof KVCache && state.kvCache.layout === 'contiguous'
    && !('modelPartition' in state && state.modelPartition) && !state.modelConfig?.hyperConnections
    && state.modelConfig?.decodeStrategy !== 'replay_prefill'
    && !(state.modelConfig?.layerTypes || []).some(isConvLayerType);
}

/** @param {import('./sequence-state.js').ContextEntry} entry */
function release(entry) {
  entry.cache?.destroy();
  resetLinearAttentionRuntime(entry.linear);
  entry.cache = null;
  entry.linear = null;
  entry.tokens = [];
  entry.bytes = 0;
  entry.signature = null;
}

/** @param {NonNullable<import('./state.js').PipelineState['kvCache']>} cache
 * @param {import('./state.js').PipelineState['linearAttentionRuntime']} linear
 * @param {number} tokenCount */
function storageBytes(cache, linear, tokenCount) {
  let bytes = cache.getMemoryStats().allocated + tokenCount * 8;
  for (const layer of linear?.layers?.values() || []) {
    for (const value of Object.values(layer)) {
      if (ArrayBuffer.isView(value)) bytes += value.byteLength;
      else if (value && typeof value.size === 'number' && typeof value.destroy === 'function') bytes += value.size;
    }
  }
  return bytes;
}

/** @type {import('./sequence-state.js').createGenerationContext} */
export function createGenerationContext(state, { maxBytes }) {
  if (!supportsGenerationContext(state)) throw new Error('Complete generation context is unsupported for this decoder.');
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error('Generation context requires a positive maxBytes.');
  /** @type {import('./sequence-state.js').ContextEntry} */
  const entry = { owner: state, maxBytes, cache: null, linear: null, tokens: [], bytes: 0,
    signature: null, busy: false, closed: false, reusedTokens: 0, status: 'empty' };
  const handle = Object.freeze({
    getStats: () => Object.freeze({ allocatedBytes: entry.bytes, cachedTokens: entry.tokens.length,
      reusedTokens: entry.reusedTokens, status: entry.status }),
    close() {
      if (entry.busy) throw new Error('Generation context cannot close before execution settles.');
      release(entry);
      entry.closed = true;
      entry.status = 'closed';
      owners.get(state)?.delete(handle);
    },
  });
  records.set(handle, entry);
  if (!owners.has(state)) owners.set(state, new Set());
  owners.get(state)?.add(handle);
  return handle;
}

/** @type {import('./sequence-state.js').closeGenerationContexts} */
export function closeGenerationContexts(state) {
  for (const handle of owners.get(state) || []) handle.close();
}

/** @type {import('./sequence-state.js').beginGenerationContext} */
export function beginGenerationContext(state, context, options) {
  if (!context) return;
  const entry = records.get(context);
  if (!entry || entry.owner !== state || entry.closed || entry.busy) throw new Error('Invalid or busy generation context.');
  if (!state.kvCache || !state.modelConfig || !supportsGenerationContext(state)) throw new Error('Generation context decoder is unavailable.');
  for (const name of /** @type {const} */ (['embeddingOverrides', '__activationTrace', 'useSpeculative'])) {
    if (options[name]) throw new Error(`Generation context does not support ${name}.`);
  }
  // Forward arithmetic and adapter identity must match. Sampling choices do not
  // change the representation of the already validated token prefix.
  const signature = JSON.stringify({ runtime: state.runtimeConfig, model: state.modelConfig,
    plan: options.executionPlan, speculation: options.speculation, adapter: state.lora?.identity?.digest ?? null });
  if (entry.signature !== signature || (state.lora && !state.lora.identity?.digest && entry.adapter !== state.lora)) release(entry);
  entry.reusedTokens = 0;
  const estimated = state.kvCache.getMemoryStats().allocated;
  if (!Number.isFinite(estimated) || estimated > entry.maxBytes) {
    release(entry);
    entry.status = 'budget';
    return;
  }
  const cache = entry.cache || createKVCache(state.modelConfig, state.useGPU, state.debug, state.runtimeConfig.inference);
  const linear = entry.linear || createLinearAttentionRuntime();
  const previous = { cache: state.kvCache, linear: state.linearAttentionRuntime, seqLen: state.currentSeqLen };
  leases.set(state, { entry, previous, signature });
  entry.busy = true;
  state.kvCache = cache;
  state.linearAttentionRuntime = linear;
  state.currentSeqLen = entry.tokens.length;
  entry.cache = null;
  entry.linear = null;
  state.wholeTokenProgram?.program.invalidate('generation context changed');
}

/** @type {import('./sequence-state.js').generationContextInput} */
export function generationContextInput(state, inputIds) {
  const lease = leases.get(state);
  if (!lease) {
    state.stats.reusedPrefixTokens = 0;
    state.stats.prefillComputedTokens = inputIds.length;
    return inputIds;
  }
  const entry = lease.entry;
  const count = entry.tokens.length;
  // Hybrid recurrent state cannot rewind to an arbitrary common prefix. Only a
  // complete match can reuse it. Equal inputs need fresh last-position logits.
  if (count > 0 && inputIds.length > count && entry.tokens.every((id, index) => id === inputIds[index])) {
    entry.reusedTokens = count;
    state.stats.reusedPrefixTokens = count;
    state.stats.prefillComputedTokens = inputIds.length - count;
    return inputIds.slice(count);
  }
  state.kvCache?.clear();
  state.linearAttentionRuntime = resetLinearAttentionSequence(state.linearAttentionRuntime);
  state.currentSeqLen = 0;
  entry.tokens = [];
  entry.reusedTokens = 0;
  state.stats.reusedPrefixTokens = 0;
  state.stats.prefillComputedTokens = inputIds.length;
  return inputIds;
}

/** @type {import('./sequence-state.js').generationContextCheckpointLength} */
export function generationContextCheckpointLength(state, prompt, inputIds, opts) {
  if (!leases.has(state)) return null;
  if (!state.tokenizer) throw new Error('Generation checkpoint requires the loaded tokenizer.');
  let end = inputIds.length - 1;
  if (!opts.inputIds && opts.useChatTemplate) {
    const messages = typeof prompt === 'string' ? [{ role: 'user', content: prompt }]
      : Array.isArray(prompt) ? prompt : prompt.messages;
    // Generation-only thinking markers need not be preserved in visible chat
    // history. Derive the shared token boundary using the actual formatter.
    const historical = state.tokenizer.encode(resolvePromptInput({ modelConfig: state.modelConfig },
      [...messages, { role: 'assistant', content: '' }], true, 'generationContext'));
    end = 0;
    while (end < inputIds.length - 1 && end < historical.length && inputIds[end] === historical[end]) end++;
  }
  return end;
}

/** @type {import('./sequence-state.js').checkpointGenerationContext} */
export async function checkpointGenerationContext(state, tokens) {
  const lease = leases.get(state);
  if (!lease || tokens.length === 0) return;
  if (!(state.kvCache instanceof KVCache) || !(lease.previous.cache instanceof KVCache)) throw new Error('Generation checkpoint requires complete KV storage.');
  // Park the checkpoint and continue on the original working bank. Copying GPU
  // state never reads it through JavaScript; the model weight banks stay shared.
  await lease.previous.cache.copyFrom(state.kvCache);
  resetLinearAttentionRuntime(lease.previous.linear);
  const linear = await cloneLinearAttentionRuntimeOnGPU(state.linearAttentionRuntime);
  lease.checkpoint = { cache: state.kvCache, linear: state.linearAttentionRuntime, tokens: [...tokens] };
  lease.previous.linear = linear;
  state.kvCache = lease.previous.cache;
  state.linearAttentionRuntime = linear;
  state.wholeTokenProgram?.program.invalidate('generation checkpoint parked');
}

/** @type {import('./sequence-state.js').prepareGenerationContextPrefill} */
export async function prepareGenerationContextPrefill(generator, prompt, inputIds, opts) {
  const state = generator._state;
  const checkpoint = generationContextCheckpointLength(state, prompt, inputIds, opts);
  if (checkpoint === null) return null;
  if (checkpoint < state.currentSeqLen) {
    // A changed template boundary cannot rewind a hybrid recurrent checkpoint.
    const lease = leases.get(state);
    if (lease) lease.entry.tokens = [];
    generationContextInput(state, inputIds);
  }
  if (checkpoint === 0) return 0;
  const chunk = generator._resolvePrefillTokenChunkSize(inputIds) ?? checkpoint;
  while (state.currentSeqLen < checkpoint) {
    const end = Math.min(checkpoint, state.currentSeqLen + chunk);
    await generator._commitPrefillHiddenChunk(await generator._prefillToHidden(inputIds.slice(state.currentSeqLen, end), opts));
  }
  await checkpointGenerationContext(state, inputIds.slice(0, checkpoint));
  return checkpoint;
}

/** @type {import('./sequence-state.js').finishGenerationContext} */
export async function finishGenerationContext(state, generatedIds, completed) {
  const lease = leases.get(state);
  if (!lease) return;
  const { entry, previous, signature } = lease;
  try {
    // Cancellation never makes already submitted commands disappear.
    if (state.useGPU) {
      const device = getDevice();
      if (!device) throw new Error('Generation context lost its GPU device.');
      await device.queue.onSubmittedWorkDone();
    }
    const checkpoint = lease.checkpoint;
    const count = checkpoint?.tokens.length ?? state.currentSeqLen;
    const valid = completed && !!checkpoint && count > 0 && count <= generatedIds.length;
    entry.cache = checkpoint?.cache ?? state.kvCache;
    if (!entry.cache) throw new Error('Generation context lost its KV storage.');
    entry.linear = checkpoint?.linear ?? state.linearAttentionRuntime;
    entry.bytes = storageBytes(entry.cache, entry.linear, count);
    if (valid && entry.bytes <= entry.maxBytes) {
      entry.tokens = [...checkpoint.tokens];
      entry.signature = signature;
      entry.adapter = state.lora;
      entry.status = 'cached';
    } else {
      entry.status = valid ? 'budget' : 'discarded';
      release(entry);
    }
  } catch (error) {
    entry.cache = lease.checkpoint?.cache ?? state.kvCache;
    entry.linear = lease.checkpoint?.linear ?? state.linearAttentionRuntime;
    release(entry);
    entry.status = 'failed';
    throw error;
  } finally {
    // Once parked, generation owns the working bank and may replace it during
    // declared precision recovery. Never restore that bank's obsolete pointer.
    if (!lease.checkpoint) {
      state.kvCache = previous.cache;
      state.linearAttentionRuntime = previous.linear;
      state.currentSeqLen = previous.seqLen;
    }
    state.wholeTokenProgram?.program.invalidate('generation context detached');
    entry.busy = false;
    leases.delete(state);
  }
}

// Explicit sequence reset and application-owned generation context memory.
/** @param {import('./sequence-state.js').SequenceState} state @param {number} seqLen */
export function resetSequenceState(state, seqLen) {
  if (state.isGenerating) {
    throw new Error('InferencePipeline.resetToSeqLen: cannot reset while generation is in progress');
  }
  if (!Number.isSafeInteger(seqLen) || seqLen < 0) {
    throw new Error('InferencePipeline.resetToSeqLen: seqLen must be a finite non-negative safe integer');
  }
  if (seqLen > state.currentSeqLen) {
    throw new Error(`InferencePipeline.resetToSeqLen: target ${seqLen} exceeds currentSeqLen ${state.currentSeqLen}`);
  }
  state.kvCache?.truncate?.(seqLen);
  state.currentSeqLen = seqLen;
}
