import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { createKVCache } from './init.js';
import { createLinearAttentionRuntime, resetLinearAttentionRuntime, resetLinearAttentionSequence } from './linear-attention.js';
import { isConvLayerType } from './layer-input-execution.js';
import { resolveLayerPartition } from './layer-partition-contract.js';
import { withHyperConnectionTransaction } from './hyper-connection/sequence-transaction.js';
import { KVCache } from '../../kv-cache/base.js';
import { cloneLinearAttentionRuntimeOnGPU } from './linear-attention/state-clone.js';
import { resolvePromptInput } from './generator/prompt-input.js';

// Handles are local to one loaded pipeline. Applications own naming, admission,
// retention and eviction; this module owns complete decoder memory and settlement.
/** @type {WeakMap<import('./sequence-state.js').GenerationContext, import('./sequence-state.js').ContextEntry>} */
const records = new WeakMap();
/** @type {WeakMap<import('./sequence-state.js').ContextPipelineState, Set<import('./sequence-state.js').GenerationContext>>} */
const owners = new WeakMap();
/** @type {WeakMap<import('./sequence-state.js').ContextPipelineState, import('./sequence-state.js').ContextLease>} */
const leases = new WeakMap();

/** @param {Record<string, unknown> | undefined} convState */
function getConvStateBuffer(convState) {
  const candidate = convState?.convStateGPU;
  return candidate && typeof (/** @type {GPUBuffer} */ (candidate)).size === 'number'
    ? /** @type {GPUBuffer} */ (candidate)
    : null;
}

/** @param {import('./sequence-state.js').ContextPipelineState} state */
function resolveContextPartition(state) {
  if (!('modelPartition' in state) || !state.modelPartition) return null;
  if (!state.manifest) return undefined;
  try {
    return resolveLayerPartition(state.manifest, state.modelPartition);
  } catch {
    return undefined;
  }
}

/** @param {Map<number, GPUBuffer> | null | undefined} conv */
function releaseConvSnapshot(conv) {
  for (const buffer of conv?.values() || []) releaseBuffer(buffer);
}

/** @param {import('./sequence-state.js').HyperConnectionSnapshot | null | undefined} hyper */
function releaseHyperSnapshot(hyper) {
  if (!hyper) return;
  for (const saved of hyper.ple.values()) releaseBuffer(saved.conv);
  for (const saved of hyper.qsa.values()) {
    if (saved.rawKeys) releaseBuffer(saved.rawKeys);
    if (saved.blockKeys) releaseBuffer(saved.blockKeys);
  }
}

/** @param {import('./state.js').PipelineState['convLayerStates']} convLayerStates */
async function snapshotConvOnGPU(convLayerStates) {
  if (!convLayerStates?.size) return null;
  const device = getDevice();
  if (!device) throw new Error('Generation checkpoint requires a live GPU device.');
  /** @type {Map<number, GPUBuffer>} */
  const snapshot = new Map();
  try {
    const encoder = device.createCommandEncoder({ label: 'generation_context_conv_snapshot' });
    for (const [index, convState] of convLayerStates) {
      const source = getConvStateBuffer(convState);
      if (!source) continue;
      const copy = acquireBuffer(source.size, undefined, `L${index}.conv_checkpoint`);
      snapshot.set(index, copy);
      encoder.copyBufferToBuffer(source, 0, copy, 0, source.size);
    }
    if (snapshot.size > 0) {
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    return snapshot;
  } catch (error) {
    releaseConvSnapshot(snapshot);
    throw error;
  }
}

/** @param {import('./state.js').PipelineState['convLayerStates']} convLayerStates
 * @param {Map<number, GPUBuffer> | null | undefined} snapshot */
async function restoreConvOnGPU(convLayerStates, snapshot) {
  if (!convLayerStates?.size || !snapshot?.size) return;
  const device = getDevice();
  if (!device) throw new Error('Generation context restore requires a live GPU device.');
  const encoder = device.createCommandEncoder({ label: 'generation_context_conv_restore' });
  for (const [index, saved] of snapshot) {
    const target = getConvStateBuffer(convLayerStates.get(index));
    if (!target || saved.size < target.size) throw new Error(`Generation context conv state mismatch at layer ${index}.`);
    encoder.copyBufferToBuffer(saved, 0, target, 0, target.size);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

/** @param {import('./state.js').PipelineState['hyperConnectionSession']} session */
async function snapshotHyperConnectionOnGPU(session) {
  if (!session) return null;
  const device = getDevice();
  if (!device) throw new Error('Generation checkpoint requires a live GPU device.');
  /** @type {import('./sequence-state.js').HyperConnectionSnapshot} */
  const snapshot = { ple: new Map(), qsa: new Map() };
  try {
    const encoder = device.createCommandEncoder({ label: 'generation_context_hyper_snapshot' });
    let copied = false;
    for (const [index, layer] of session.layers) {
      const conv = acquireBuffer(layer.conv.buffer.size, undefined, `L${index}.ple_context_conv`);
      snapshot.ple.set(index, { conv });
      encoder.copyBufferToBuffer(layer.conv.buffer, 0, conv, 0, conv.size);
      copied = true;
    }
    if (session.qsa) {
      const rowBytes = session.qsa.contract.headDim * 4;
      const compressRatio = session.qsa.contract.compressRatio;
      for (const [index, layer] of session.qsa.layers) {
        const rawBytes = layer.length * rowBytes;
        const blockBytes = Math.floor(layer.length / compressRatio) * rowBytes;
        const rawKeys = rawBytes > 0 ? acquireBuffer(rawBytes, undefined, `L${index}.qsa_context_raw`) : null;
        const blockKeys = blockBytes > 0 ? acquireBuffer(blockBytes, undefined, `L${index}.qsa_context_block`) : null;
        snapshot.qsa.set(index, { length: layer.length, rawKeys, blockKeys });
        if (rawKeys) {
          encoder.copyBufferToBuffer(layer.rawKeys, 0, rawKeys, 0, rawBytes);
          copied = true;
        }
        if (blockKeys) {
          encoder.copyBufferToBuffer(layer.blockKeys, 0, blockKeys, 0, blockBytes);
          copied = true;
        }
      }
    }
    if (copied) {
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    return snapshot;
  } catch (error) {
    releaseHyperSnapshot(snapshot);
    throw error;
  }
}

/** @param {import('./state.js').PipelineState['hyperConnectionSession']} session
 * @param {import('./sequence-state.js').HyperConnectionSnapshot | null | undefined} snapshot
 * @param {readonly number[]} tokens */
async function restoreHyperConnectionOnGPU(session, snapshot, tokens) {
  if (!session || !snapshot) return;
  const device = getDevice();
  if (!device) throw new Error('Generation context restore requires a live GPU device.');
  const encoder = device.createCommandEncoder({ label: 'generation_context_hyper_restore' });
  let copied = false;
  for (const [index, saved] of snapshot.ple) {
    const layer = session.layers.get(index);
    const bytes = layer?.conv.buffer.size ?? 0;
    if (!layer || !bytes || saved.conv.size < bytes) {
      throw new Error(`Generation context PLE state mismatch at layer ${index}.`);
    }
    encoder.copyBufferToBuffer(saved.conv, 0, layer.conv.buffer, 0, bytes);
    layer.history.reset();
    if (tokens.length > 0) layer.history.append(tokens);
    copied = true;
  }
  if (session.qsa) {
    const rowBytes = session.qsa.contract.headDim * 4;
    const compressRatio = session.qsa.contract.compressRatio;
    for (const [index, saved] of snapshot.qsa) {
      const layer = session.qsa.layers.get(index);
      const rawBytes = saved.length * rowBytes;
      const blockBytes = Math.floor(saved.length / compressRatio) * rowBytes;
      if (!layer || saved.length > session.qsa.capacity || saved.length !== tokens.length) {
        throw new Error(`Generation context QSA state mismatch at layer ${index}.`);
      }
      if (rawBytes > 0) {
        if (!saved.rawKeys || saved.rawKeys.size < rawBytes || layer.rawKeys.size < rawBytes) {
          throw new Error(`Generation context QSA raw state mismatch at layer ${index}.`);
        }
        encoder.copyBufferToBuffer(saved.rawKeys, 0, layer.rawKeys, 0, rawBytes);
        copied = true;
      }
      if (blockBytes > 0) {
        if (!saved.blockKeys || saved.blockKeys.size < blockBytes || layer.blockKeys.size < blockBytes) {
          throw new Error(`Generation context QSA block state mismatch at layer ${index}.`);
        }
        encoder.copyBufferToBuffer(saved.blockKeys, 0, layer.blockKeys, 0, blockBytes);
        copied = true;
      }
      layer.length = saved.length;
      layer.reservedEnd = saved.length;
    }
    session.qsa.epoch += 1;
  }
  if (copied) {
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
}

/** @type {import('./sequence-state.js').supportsGenerationContext} */
export function supportsGenerationContext(state) {
  if (state.useGPU !== true || !(state.kvCache instanceof KVCache) || state.kvCache.layout !== 'contiguous') {
    return false;
  }
  if (state.modelConfig?.decodeStrategy === 'replay_prefill') {
    return false;
  }
  const partition = resolveContextPartition(state);
  if (partition === undefined) {
    return false;
  }
  if (state.modelConfig?.hyperConnections && !state.hyperConnectionSession) {
    return false;
  }
  const layerTypes = state.modelConfig?.layerTypes || [];
  const [start, end] = partition ? partition.layerRange : [0, layerTypes.length - 1];
  for (let i = start; i <= end && i < layerTypes.length; i++) {
    if (isConvLayerType(layerTypes[i]) && !getConvStateBuffer(state.convLayerStates?.get(i))) {
      return false;
    }
  }
  return true;
}

/** @param {import('./sequence-state.js').ContextEntry} entry */
function release(entry) {
  entry.cache?.destroy();
  resetLinearAttentionRuntime(entry.linear);
  releaseConvSnapshot(entry.conv);
  releaseHyperSnapshot(entry.hyper);
  entry.cache = null;
  entry.linear = null;
  entry.conv = null;
  entry.hyper = null;
  entry.tokens = [];
  entry.bytes = 0;
  entry.signature = null;
}

/** @param {NonNullable<import('./state.js').PipelineState['kvCache']>} cache
 * @param {import('./state.js').PipelineState['linearAttentionRuntime']} linear
 * @param {Map<number, GPUBuffer> | null} conv
 * @param {import('./sequence-state.js').HyperConnectionSnapshot | null} hyper
 * @param {number} tokenCount */
function storageBytes(cache, linear, conv, hyper, tokenCount) {
  let bytes = cache.getMemoryStats().allocated + tokenCount * 8;
  for (const layer of linear?.layers?.values() || []) {
    for (const value of Object.values(layer)) {
      if (ArrayBuffer.isView(value)) bytes += value.byteLength;
      else if (value && typeof value.size === 'number' && typeof value.destroy === 'function') bytes += value.size;
    }
  }
  for (const buffer of conv?.values() || []) bytes += buffer.size;
  for (const saved of hyper?.ple.values() || []) bytes += saved.conv.size;
  for (const saved of hyper?.qsa.values() || []) {
    bytes += (saved.rawKeys?.size ?? 0) + (saved.blockKeys?.size ?? 0);
  }
  return bytes;
}

/** @type {import('./sequence-state.js').createGenerationContext} */
export function createGenerationContext(state, { maxBytes }) {
  if (!supportsGenerationContext(state)) throw new Error('Complete generation context is unsupported for this decoder.');
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error('Generation context requires a positive maxBytes.');
  /** @type {import('./sequence-state.js').ContextEntry} */
  const entry = { owner: state, maxBytes, cache: null, linear: null, conv: null, hyper: null, tokens: [], bytes: 0,
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
  const partition = resolveContextPartition(state);
  // Forward arithmetic and adapter identity must match. Sampling choices do not
  // change the representation of the already validated token prefix.
  const signature = JSON.stringify({ runtime: state.runtimeConfig, model: state.modelConfig,
    partition: state.modelPartition ?? null,
    plan: options.executionPlan, speculation: options.speculation, adapter: state.lora?.identity?.digest ?? null });
  if (entry.signature !== signature || (state.lora && !state.lora.identity?.digest && entry.adapter !== state.lora)) release(entry);
  entry.reusedTokens = 0;
  const estimated = state.kvCache.getMemoryStats().allocated;
  if (!Number.isFinite(estimated) || estimated > entry.maxBytes) {
    release(entry);
    entry.status = 'budget';
    return;
  }
  /** @type {readonly [number, number] | null} */
  const layerRange = partition ? [partition.layerRange[0], partition.layerRange[1]] : null;
  const cache = entry.cache || createKVCache(state.modelConfig, state.useGPU, state.debug, state.runtimeConfig.inference, layerRange);
  const linear = entry.linear || createLinearAttentionRuntime();
  const previous = { cache: state.kvCache, linear: state.linearAttentionRuntime, seqLen: state.currentSeqLen };
  leases.set(state, { entry, previous, signature, conv: entry.conv, hyper: entry.hyper });
  entry.busy = true;
  state.kvCache = cache;
  state.linearAttentionRuntime = linear;
  state.currentSeqLen = entry.tokens.length;
  if (entry.tokens.length === 0) state.hyperConnectionSession?.reset();
  entry.cache = null;
  entry.linear = null;
  entry.conv = null;
  entry.hyper = null;
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
  state.hyperConnectionSession?.reset();
  releaseConvSnapshot(lease.conv);
  releaseHyperSnapshot(lease.hyper);
  lease.conv = null;
  lease.hyper = null;
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
  const conv = await snapshotConvOnGPU(state.convLayerStates);
  const hyper = await snapshotHyperConnectionOnGPU(state.hyperConnectionSession);
  lease.checkpoint = { cache: state.kvCache, linear: state.linearAttentionRuntime, conv, hyper, tokens: [...tokens] };
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
  const lease = leases.get(state);
  if (checkpoint < state.currentSeqLen) {
    // A changed template boundary cannot rewind a hybrid recurrent checkpoint.
    if (lease) lease.entry.tokens = [];
    generationContextInput(state, inputIds);
  }
  if (checkpoint === 0) return 0;
  if (state.currentSeqLen > 0 && lease) {
    if (lease.conv) {
      await restoreConvOnGPU(state.convLayerStates, lease.conv);
      releaseConvSnapshot(lease.conv);
      lease.conv = null;
    }
    if (lease.hyper) {
      await restoreHyperConnectionOnGPU(state.hyperConnectionSession, lease.hyper, lease.entry.tokens);
      releaseHyperSnapshot(lease.hyper);
      lease.hyper = null;
    }
  }
  const chunk = generator._resolvePrefillTokenChunkSize(inputIds) ?? checkpoint;
  if (state.currentSeqLen < checkpoint) {
    await withHyperConnectionTransaction(state, opts, async (scoped) => {
      while (state.currentSeqLen < checkpoint) {
        const end = Math.min(checkpoint, state.currentSeqLen + chunk);
        const hidden = await generator._prefillToHidden(
          inputIds.slice(state.currentSeqLen, end),
          /** @type {import('./sequence-state.js').ContextOptions} */ (scoped)
        );
        await generator._commitPrefillHiddenChunk(hidden);
      }
    });
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
    entry.conv = checkpoint?.conv ?? lease.conv ?? null;
    entry.hyper = checkpoint?.hyper ?? lease.hyper ?? null;
    lease.conv = null;
    lease.hyper = null;
    entry.bytes = storageBytes(entry.cache, entry.linear, entry.conv, entry.hyper, count);
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
    entry.conv = lease.checkpoint?.conv ?? lease.conv ?? null;
    entry.hyper = lease.checkpoint?.hyper ?? lease.hyper ?? null;
    lease.conv = null;
    lease.hyper = null;
    release(entry);
    entry.status = 'failed';
    throw error;
  } finally {
    releaseConvSnapshot(lease.conv);
    releaseHyperSnapshot(lease.hyper);
    // Once parked, generation owns the working bank and may replace it during
    // declared precision recovery. Never restore that bank's obsolete pointer.
    if (!lease.checkpoint) {
      state.kvCache = previous.cache;
      state.linearAttentionRuntime = previous.linear;
      state.currentSeqLen = previous.seqLen;
      state.hyperConnectionSession?.reset();
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
