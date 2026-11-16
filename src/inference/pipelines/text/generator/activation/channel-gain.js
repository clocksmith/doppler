import { channelEnergy, channelEnergyPair, channelGainSwiGLU, validateChannelTensor } from "../../../../../gpu/kernels/channel-gain.js";
import { deriveChannelRotation, applyChannelRotation } from "../../../../../gpu/kernels/channel-rotation.js";
import { copyReplayTensor, replayFinalFFN } from './final-layer-replay.js';
import { createTensor } from "../../../../../gpu/tensor.js";
import { createSha256Hasher } from "../../../../../formats/sha256.js";
import { getDevice } from "../../../../../gpu/device.js";
import { acquireBuffer, readBuffer, releaseBuffer } from "../../../../../memory/buffer-pool.js";



/** @type {WeakSet<object>} */
const activeOperations = new WeakSet();

/** @type {typeof import('./channel-gain.js').assertChannelGainIdle} */
export function assertChannelGainIdle(state) {
  if (state.isGenerating || activeOperations.has(state)) throw new Error('Cannot change channel gains during an active operation.');
}

/** @type {typeof import('./channel-gain.js').withChannelGainOperation} */
export async function withChannelGainOperation(state, operation) {
  if (activeOperations.has(state)) throw new Error('Incremental channel-gain operations must be serialized.');
  activeOperations.add(state);
  try { return await operation(); }
  finally { activeOperations.delete(state); }
}

/** @type {WeakMap<import('./channel-gain.js').FrozenChannelGains, { layers: Map<number, { channels: number; gains?: import("../../../../../gpu/tensor.js").Tensor; rotation?: import("../../../../../gpu/kernels/channel-rotation.js").ChannelRotation; theta?: number }>; kind: 'gain' | 'rotation'; users: number; closed: boolean }>} */
const frozenStates = new WeakMap();

/** @type {typeof import('./channel-gain.js').freezeChannelGains} */
export async function freezeChannelGains(layers) {
  const device = getDevice();
  if (!device || !Array.isArray(layers) || !layers.length) throw new Error('Freezing gains requires a device and selected layers.');
  const state = { layers: new Map(), kind: /** @type {const} */ ('gain'), users: 0, closed: false };
  const hasher = createSha256Hasher();
  try {
    for (const layer of [...layers].sort((a, b) => a.layerIdx - b.layerIdx)) {
      if (!Number.isInteger(layer.layerIdx) || layer.layerIdx < 0 || state.layers.has(layer.layerIdx)) {
        throw new Error('Frozen gains require unique nonnegative layer indices.');
      }
      validateChannelTensor(layer.gains, 1, layer.channels);
      const copy = acquireBuffer(layer.channels * 4, undefined, 'frozen_channel_gains');
      state.layers.set(layer.layerIdx, { channels: layer.channels,
        gains: createTensor(copy, 'f32', [layer.channels], 'frozen_channel_gains') });
      const encoder = device.createCommandEncoder({ label: 'freeze_channel_gains' });
      encoder.copyBufferToBuffer(layer.gains.buffer, 0, copy, 0, layer.channels * 4);
      device.queue.submit([encoder.finish()]);
      const bytes = await readBuffer(copy, layer.channels * 4);
      if (new Float32Array(bytes).some(value => !Number.isFinite(value) || value <= 0)) {
        throw new Error('Frozen channel gains must all be finite and positive.');
      }
      hasher.update(new TextEncoder().encode(`channel-gain-f32-v1:${layer.layerIdx}:${layer.channels}:`));
      hasher.update(new Uint8Array(bytes));
    }
    const frozen = Object.freeze({
      gainHash: hasher.digestHex(),
      gpuBytes: [...state.layers.values()].reduce((sum, layer) => sum + layer.gains.buffer.size, 0),
      logicalBytes: [...state.layers.values()].reduce((sum, layer) => sum + layer.channels * 4, 0),
      layers: Object.freeze([...state.layers].map(([layerIdx, layer]) => Object.freeze({ layerIdx, channels: layer.channels }))),
      close() {
        if (state.closed) return;
        if (state.users) throw new Error('Close active gain requests before releasing frozen gains.');
        state.closed = true;
        const buffers = [...state.layers.values()].map(layer => layer.gains.buffer);
        state.layers.clear();
        const release = () => { for (const buffer of buffers) releaseBuffer(buffer); };
        device.queue.onSubmittedWorkDone().then(release, release);
      },
    });
    frozenStates.set(frozen, state);
    return frozen;
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => {});
    for (const layer of state.layers.values()) releaseBuffer(layer.gains.buffer);
    throw error;
  }
}

/** @type {typeof import('./channel-gain.js').freezeChannelRotations} */
export async function freezeChannelRotations(layers) {
  const device = getDevice();
  if (!device || !Array.isArray(layers) || !layers.length) throw new Error('Freezing rotations requires a device and selected layers.');
  const state = { layers: new Map(), kind: /** @type {const} */ ('rotation'), users: 0, closed: false };
  const hasher = createSha256Hasher();
  /** @type {GPUBuffer[]} */
  const buffers = [];
  try {
    for (const layer of [...layers].sort((a, b) => a.layerIdx - b.layerIdx)) {
      if (!Number.isInteger(layer.layerIdx) || layer.layerIdx < 0 || state.layers.has(layer.layerIdx)
          || !Number.isFinite(layer.theta) || !Number.isFinite(Math.fround(layer.theta))) {
        throw new Error('Frozen rotations require unique layers and finite f32 angles.');
      }
      validateChannelTensor(layer.rotation.basis, 2, layer.channels);
      validateChannelTensor(layer.rotation.status, 1, 1);
      const basis = acquireBuffer(layer.channels * 8, undefined, 'frozen_rotation_basis');
      buffers.push(basis);
      const status = acquireBuffer(4, undefined, 'frozen_rotation_status');
      buffers.push(status);
      const encoder = device.createCommandEncoder({ label: 'freeze_channel_rotation' });
      encoder.copyBufferToBuffer(layer.rotation.basis.buffer, 0, basis, 0, layer.channels * 8);
      encoder.copyBufferToBuffer(layer.rotation.status.buffer, 0, status, 0, 4);
      device.queue.submit([encoder.finish()]);
      const bytes = await readBuffer(basis, layer.channels * 8);
      const statusBytes = await readBuffer(status, 4);
      const value = new Float32Array(statusBytes)[0];
      const plane = new Float32Array(bytes);
      let uu = 0; let vv = 0; let uv = 0;
      for (let i = 0; i < layer.channels; i += 1) {
        uu += plane[i] * plane[i]; vv += plane[i + layer.channels] ** 2;
        uv += plane[i] * plane[i + layer.channels];
      }
      if (plane.some(x => !Number.isFinite(x)) || ![0, 1].includes(value)
          || (value === 0 && plane.some(x => x !== 0))
          || (value === 1 && (Math.abs(uu - 1) > 2e-5 || Math.abs(vv - 1) > 2e-5 || Math.abs(uv) > 2e-5))) {
        throw new Error('Frozen rotation must be an orthonormal plane or explicit degenerate identity.');
      }
      const theta = Math.fround(layer.theta);
      hasher.update(new TextEncoder().encode(`channel-rotation-f32-v1:${layer.layerIdx}:${layer.channels}:${theta}:`));
      hasher.update(new Uint8Array(bytes)); hasher.update(new Uint8Array(statusBytes));
      state.layers.set(layer.layerIdx, { channels: layer.channels, theta, rotation: {
        basis: createTensor(basis, 'f32', [2, layer.channels]), status: createTensor(status, 'f32', [1]),
      } });
    }
    const frozen = Object.freeze({ gainHash: hasher.digestHex(),
      gpuBytes: buffers.reduce((sum, buffer) => sum + buffer.size, 0),
      logicalBytes: layers.reduce((sum, layer) => sum + layer.channels * 8 + 4, 0),
      layers: Object.freeze([...state.layers].map(([layerIdx, layer]) => Object.freeze({ layerIdx, channels: layer.channels }))),
      close() {
        if (state.closed) return;
        if (state.users) throw new Error('Close active rotation requests before releasing frozen rotations.');
        state.closed = true; state.layers.clear();
        const release = () => { for (const buffer of buffers) releaseBuffer(buffer); };
        device.queue.onSubmittedWorkDone().then(release, release);
      },
    });
    frozenStates.set(frozen, state);
    return frozen;
  } catch (error) {
    await device.queue.onSubmittedWorkDone().catch(() => {});
    for (const buffer of buffers) releaseBuffer(buffer);
    throw error;
  }
}

/** @type {typeof import('./channel-gain.js').createChannelGainRequest} */
export function createChannelGainRequest(spec) {
  const { mode, requestId, gainHash, cacheArm, tokenCount, span, layers, frozenGains } = spec;
  const frozenState = frozenGains ? frozenStates.get(frozenGains) : null;
  const kind = spec.transformKind ?? 'gain';
  if (!['gain', 'rotation'].includes(kind) || (frozenState && frozenState.kind !== kind)) {
    throw new Error('Research transform kind must match its opaque frozen handle.');
  }
  if (!['capture', 'apply'].includes(mode) || !requestId || !['base-cache', 're-prefill'].includes(cacheArm)
      || !Number.isSafeInteger(tokenCount) || tokenCount < 1 || !Array.isArray(layers) || layers.length === 0) {
    throw new Error('Channel gain request requires explicit mode, requestId, cacheArm, tokenCount and selected layers.');
  }
  if (!span || !Number.isInteger(span.start) || !Number.isInteger(span.stop)
      || span.start < 0 || span.stop > tokenCount || span.start >= span.stop
      || !Array.isArray(span.templatePositions) || span.templatePositions.some(p =>
        !Number.isInteger(p) || p < 0 || p >= tokenCount || (p >= span.start && p < span.stop))) {
    throw new Error('Channel gain requires exact nonempty instruction bounds excluding all declared template positions.');
  }
  if (mode === 'apply' && (!frozenState || frozenState.closed || gainHash !== frozenGains?.gainHash)) {
    throw new Error('Applied gains require a live opaque frozen gain handle and its verified SHA-256 gainHash.');
  }
  /** @type {Map<number, Readonly<import('./channel-gain.js').ChannelGainSpec['layers'][number]>>} */
  const selected = new Map();
  for (const layer of layers) {
    if (!Number.isInteger(layer.layerIdx) || layer.layerIdx < 0 || selected.has(layer.layerIdx)
        || !Number.isInteger(layer.channels) || layer.channels < 1) {
      throw new Error('Channel gain layers require unique nonnegative indices and positive explicit channels.');
    }
    if (mode === 'apply') {
      if (frozenState?.layers.get(layer.layerIdx)?.channels !== layer.channels) {
        throw new Error('Selected layer geometry disagrees with frozen gains.');
      }
    }
    selected.set(layer.layerIdx, Object.freeze({ ...layer }));
  }
  const content = spec.captureContentSpan;
  if (content !== undefined && (mode !== 'capture'
      || !Number.isInteger(content.start) || !Number.isInteger(content.stop)
      || content.start < 0 || content.stop > tokenCount || content.start > span.start
      || content.stop < span.stop || content.stop - content.start <= span.stop - span.start
      || span.templatePositions.some(p => p >= content.start && p < content.stop))) {
    throw new Error('Same-pass capture requires content containing the instruction plus a nonempty template-free complement.');
  }
  if (kind === 'rotation' && mode === 'capture' && (!content
      || !Number.isFinite(spec.rotationEpsilon) || Math.fround(spec.rotationEpsilon ?? 0) <= 0
      || !Number.isFinite(Math.fround(spec.rotationEpsilon ?? 0)))) {
    throw new Error('Rotation capture requires an explicit content complement and finite positive f32 epsilon.');
  }
  const rotationEpsilon = spec.rotationEpsilon;
  const rotations = new Map();
  /** @type {import('./final-layer-replay.js').FinalLayerReplayCapture | null} */
  let finalReplay = null;
  const retainFinalLayerReplay = spec.retainFinalLayerReplay === true;
  if (retainFinalLayerReplay && (kind !== 'rotation' || mode !== 'capture' || selected.size !== 1)) {
    throw new Error('Final-layer replay capture requires exactly one rotation layer.');
  }
  const frozenContent = content ? Object.freeze({ start: content.start, stop: content.stop }) : null;
  const frozenSpan = Object.freeze({ start: span.start, stop: span.stop });
  /** @type {Map<number, import("../../../../../gpu/tensor.js").Tensor>} */
  const energies = new Map();
  /** @type {Map<number, import("../../../../../gpu/tensor.js").Tensor>} */
  const referenceEnergies = new Map();
  const seen = new Set();
  let closed = false;
  let pending = 0;
  const assertOpen = () => { if (closed) throw new Error('Channel gain request is closed.'); };
  /** @type {import('./channel-gain.js').ChannelGainRequest} */
  const request = {
    requestId, mode, gainHash, cacheArm, transformKind: kind,
    has(layerIdx) { assertOpen(); return selected.has(layerIdx); },
    validateLayer(layerIdx, context, tokens, channels) {
      assertOpen();
      const layer = selected.get(layerIdx);
      if (!layer) return;
      if (context.lora || context.config.useMoE || context.config.gatedActivation !== true
          || context.config.hiddenActivation !== 'silu' || context.activationPrecision === 'two-f32'
          || channels !== layer.channels || context.activationDtype !== 'f32') {
        throw new Error('Channel gain supports only explicit dense, gated, ordinary f32 SwiGLU.');
      }
      if (context.phase === 'prefill') {
        if (mode === 'apply' && cacheArm === 'base-cache') {
          throw new Error('Base-cache gains must attach after the unmodified prefill.');
        }
        if (tokens !== tokenCount || context.currentSeqLen !== 0 || seen.has(layerIdx)) {
          throw new Error('Channel gain requires one fresh full prefill; chunking, prefix reuse and repeated capture are unsupported.');
        }
      } else if (context.phase !== 'decode' || tokens !== 1 || mode !== 'apply') {
        throw new Error('Channel gain decode requires frozen gains and one token.');
      }
    },
    async activate(layerIdx, up, gate, tokens, channels, swigluLimit, recorder) {
      assertOpen();
      const layer = selected.get(layerIdx);
      const gains = frozenState?.layers.get(layerIdx)?.gains;
      if (!layer || !gains || mode !== 'apply') throw new Error('No frozen channel gains selected for activation.');
      pending += 1;
      try {
        const output = await channelGainSwiGLU(recorder, up, gate, gains,
          { tokens, channels, swigluLimit });
        seen.add(layerIdx);
        return output;
      } finally { pending -= 1; }
    },
    async transform(layerIdx, tensor, tokens, channels, recorder) {
      assertOpen();
      const layer = frozenState?.layers.get(layerIdx);
      if (kind !== 'rotation' || mode !== 'apply' || !layer?.rotation) throw new Error('No frozen rotation selected.');
      pending += 1;
      try {
        const output = await applyChannelRotation(recorder, tensor, layer.rotation, { tokens, channels, theta: layer.theta ?? NaN });
        seen.add(layerIdx);
        return output;
      } finally { pending -= 1; }
    },
    async capture(layerIdx, tensor, tokens, channels, recorder) {
      assertOpen();
      if (!selected.has(layerIdx) || mode !== 'capture') return;
      if (seen.has(layerIdx)) throw new Error('Channel energy already captured for this layer.');
      pending += 1;
      try {
        if (kind === 'rotation' && frozenContent && rotationEpsilon !== undefined) {
          rotations.set(layerIdx, await deriveChannelRotation(recorder, tensor, { tokens, channels, ...frozenSpan,
            contentStart: frozenContent.start, contentStop: frozenContent.stop, epsilon: rotationEpsilon }));
        } else if (frozenContent) {
          const pair = await channelEnergyPair(recorder, tensor, { tokens, channels, ...frozenSpan,
            contentStart: frozenContent.start, contentStop: frozenContent.stop });
          energies.set(layerIdx, pair.instruction);
          referenceEnergies.set(layerIdx, pair.reference);
        } else {
          const energy = await channelEnergy(recorder, tensor, { tokens, channels, ...frozenSpan });
          energies.set(layerIdx, energy);
        }
        seen.add(layerIdx);
      } finally { pending -= 1; }
    },
    captureFinalLayer(layerIdx, postAttn, residual, tokens, context, weights, supported) {
      assertOpen();
      if (!retainFinalLayerReplay || !selected.has(layerIdx)) return;
      if (!supported || context.phase !== 'prefill' || layerIdx !== context.config.numLayers - 1
          || tokens !== tokenCount || finalReplay) throw new Error('Unsupported or repeated final-layer replay capture.');
      const copy = copyReplayTensor(postAttn, context.recorder ?? null);
      try {
        finalReplay = { layerIdx, tokens, postAttn: copy,
          residual: residual ? copyReplayTensor(residual, context.recorder ?? null) : null,
          context: { ...context }, weights };
      } catch (error) { releaseBuffer(copy.buffer); throw error; }
    },
    async replayFinalLayer(request) {
      assertOpen();
      if (!finalReplay || pending || request.transformKind !== 'rotation' || request.mode !== 'apply'
          || request.cacheArm !== 're-prefill') throw new Error('Final-layer replay needs settled capture and a fresh re-prefill rotation request.');
      pending += 1;
      try { return await replayFinalFFN(finalReplay, request); }
      finally { pending -= 1; }
    },
    takeRotations() {
      assertOpen();
      if (kind !== 'rotation' || mode !== 'capture' || pending || rotations.size !== selected.size) {
        throw new Error('Rotation capture is absent or incomplete; settle prefill before transferring outputs.');
      }
      const result = new Map(rotations); rotations.clear(); return result;
    },
    takeEnergies() {
      assertOpen();
      if (mode !== 'capture' || pending || energies.size !== selected.size) {
        throw new Error('Channel energy capture is incomplete; settle the prefill before transferring outputs.');
      }
      const result = new Map(energies);
      energies.clear();
      return result;
    },
    takeReferenceEnergies() {
      assertOpen();
      if (!frozenContent || mode !== 'capture' || pending || referenceEnergies.size !== selected.size) {
        throw new Error('Same-pass reference capture is absent or incomplete.');
      }
      const result = new Map(referenceEnergies);
      referenceEnergies.clear();
      return result;
    },
    close() {
      if (closed) return;
      if (pending) throw new Error('Cannot close channel gain request during an active dispatch.');
      closed = true;
      if (mode === 'apply' && frozenState) frozenState.users -= 1;
      const buffers = [...energies.values(), ...referenceEnergies.values()].map(tensor => tensor.buffer);
      for (const rotation of rotations.values()) buffers.push(rotation.basis.buffer, rotation.status.buffer);
      rotations.clear();
      if (finalReplay) {
        buffers.push(finalReplay.postAttn.buffer);
        if (finalReplay.residual) buffers.push(finalReplay.residual.buffer);
        finalReplay = null;
      }
      energies.clear();
      referenceEnergies.clear();
      selected.clear();
      seen.clear();
      if (buffers.length) {
        const release = () => { for (const buffer of buffers) releaseBuffer(buffer); };
        const device = getDevice();
        if (device) device.queue.onSubmittedWorkDone().then(release, release);
        else release();
      }
    },
  };
  if (mode === 'apply' && frozenState) frozenState.users += 1;
  return Object.freeze(request);
}
