import { peekUniformCacheBytes } from '../../../../gpu/uniform-cache.js';
import { observeBufferRelease, isBufferActive, isPersistentBuffer } from '../../../../memory/buffer-pool.js';
import { log } from '../../../../debug/index.js';

// Whole-token decode program (runtime.inference.session.decodeLoop.wholeTokenProgram).
//
// The ordinary recorded decode step is encoded four times (decode steps 2-5 of a
// generation) while every GPU command it records is observed: dispatches, bind
// group descriptors, uniform contents, queue writes and copies. The three
// observations must agree on structure (pipelines, workgroup counts, bound
// buffers). Each uniform or written u32 word that differs is classified as the
// step's input token id or as an affine function of the position; every later
// observation must match the prediction made from the first two. Any other
// difference rejects the program with an error (fail closed).
//
// The accepted program owns its bind groups, one packed uniform buffer and
// private copies of the step's transient buffers, all built once. A replayed
// token writes its position-dependent uniform words and token id with queue
// writes, submits one command buffer (one compute pass between copies) and maps
// back only the sampled token id. The next token's command buffer is encoded
// while the GPU runs the current one. The math is the recorded step's: the same
// pipelines, buffers and dispatch order.

/** @typedef {import('../../../../gpu/command-recorder.js').CommandRecorder} CommandRecorder */

/**
 * A u32 word as a function of the decode position and input token.
 * @typedef {{ kind: 'const', value: number } | { kind: 'token' } | { kind: 'affine', base: number, slope: number, pos0: number }} WordModel
 */

/** @typedef {{ word: number, model: WordModel }} DynamicWord */

/**
 * A buffer as recorded in the template: the buffer itself, or `transient:<n>` for a
 * per-step transient that the program replaces with a private copy.
 * @typedef {GPUBuffer | string} BufferKey
 */

/** @typedef {{ binding: number, resource: GPUBindingResource }} BindGroupEntryRecord */
/** @typedef {{ layout: GPUBindGroupLayout, label: string | null, entries: BindGroupEntryRecord[] }} BindGroupRecord */
/** @typedef {WeakMap<GPUBindGroup, BindGroupRecord>} BindGroupDescriptors */

/**
 * One bind group entry as observed in a decode step.
 * @typedef {{ binding: number, other: unknown }
 *   | { binding: number, uniform: Uint8Array | null }
 *   | { binding: number, buffer: GPUBuffer, offset: number, size: number | null }} ObservedEntry
 */

/**
 * One bind group as observed in a decode step: opaque when created outside observation.
 * @typedef {{ opaque: GPUBindGroup }
 *   | { layout: GPUBindGroupLayout, label: string | null, entries: ObservedEntry[] }} ObservedGroup
 */

/** @typedef {{ kind: 'dispatch', pipeline: GPUComputePipeline | null, groups: ObservedGroup[], workgroups: number[] }} DispatchOp */
/** @typedef {{ kind: 'indirect', pipeline: GPUComputePipeline | null, groups: ObservedGroup[], buffer: GPUBuffer, offset: number }} IndirectOp */
/** @typedef {{ kind: 'write', buffer: GPUBuffer, offset: number, bytes: Uint8Array }} WriteOp */
/** @typedef {{ kind: 'copy', src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number }} CopyOp */
/** @typedef {{ kind: 'clear', buffer: GPUBuffer, offset: number, size: number }} ClearOp */
/** @typedef {DispatchOp | IndirectOp | WriteOp | CopyOp | ClearOp} ObservedOp */

/**
 * Field view of one observed op across the decode steps while building the template.
 * The builder checks that every step records the same kind at an index, then reads
 * only that kind's fields.
 * @typedef {{
 *   kind: ObservedOp['kind'],
 *   pipeline: GPUComputePipeline,
 *   groups: ObservedGroupView[],
 *   workgroups: number[],
 *   buffer: GPUBuffer,
 *   offset: number,
 *   bytes: Uint8Array,
 *   src: GPUBuffer,
 *   srcOffset: number,
 *   dst: GPUBuffer,
 *   dstOffset: number,
 *   size: number,
 * }} ObservedOpView
 */

/**
 * @typedef {{
 *   opaque?: GPUBindGroup,
 *   layout: GPUBindGroupLayout,
 *   label: string | null,
 *   entries: ObservedEntryView[],
 * }} ObservedGroupView
 */

/**
 * @typedef {{
 *   binding: number,
 *   other?: unknown,
 *   uniform?: Uint8Array | null,
 *   buffer: GPUBuffer,
 *   offset: number,
 *   size: number | null,
 * }} ObservedEntryView
 */

/** @typedef {{ bytes: Uint8Array, dynamic: DynamicWord[], offset: number }} UniformSlot */

/**
 * @typedef {{ binding: number, slot: UniformSlot, key?: undefined, offset?: undefined, size?: undefined }
 *   | { binding: number, slot?: undefined, key: BufferKey, offset: number, size: number | null }} TemplateEntry
 */

/**
 * @typedef {{ opaque: GPUBindGroup, layout?: undefined, label?: undefined, entries?: undefined }
 *   | { opaque?: undefined, layout: GPUBindGroupLayout, label: string | null, entries: TemplateEntry[] }} TemplateGroup
 */

/**
 * @typedef {{
 *   kind: 'dispatch' | 'indirect',
 *   pipeline: GPUComputePipeline,
 *   groups: (TemplateGroup | null)[],
 *   workgroups?: number[],
 *   indirect?: BufferKey,
 *   indirectOffset?: number,
 * }} DispatchTemplate
 */
/** @typedef {{ kind: 'write', key: BufferKey, offset: number, bytes: Uint8Array, dynamic: DynamicWord[] }} WriteTemplate */
/** @typedef {{ kind: 'copy', src: BufferKey, srcOffset: WordModel, dst: BufferKey, dstOffset: WordModel, size: number }} CopyTemplate */
/** @typedef {{ kind: 'clear', key: BufferKey, offset: number, size: number }} ClearTemplate */
/** @typedef {DispatchTemplate | WriteTemplate | CopyTemplate | ClearTemplate} TemplateStep */

/** @typedef {{ src: BufferKey, srcOffset: number, dstOffset: number, size: number }} ReadbackCopyTemplate */
/** @typedef {{ copies: ReadbackCopyTemplate[], dst: GPUBuffer[] }} ReadbackTemplate */
/** @typedef {{ size: number, usage: number }} TransientInfo */

/** @typedef {{ kind: 'copy', src: GPUBuffer, srcOffset: WordModel, dst: GPUBuffer, dstOffset: WordModel, size: number }} ReplayCopy */
/** @typedef {{ kind: 'clear', buffer: GPUBuffer, offset: number, size: number }} ReplayClear */
/** @typedef {{ kind: 'dispatch', pipeline: GPUComputePipeline, groups: (GPUBindGroup | null)[], workgroups: number[] }} ReplayDispatch */
/** @typedef {{ kind: 'indirect', pipeline: GPUComputePipeline, groups: (GPUBindGroup | null)[], indirect: GPUBuffer, indirectOffset: number }} ReplayIndirect */
/** @typedef {ReplayCopy | ReplayClear | ReplayDispatch | ReplayIndirect} ReplayStep */
/** @typedef {{ buffer: GPUBuffer, offset: number, bytes: Uint8Array, dynamic: DynamicWord[] }} ReplayWrite */
/** @typedef {{ src: GPUBuffer, srcOffset: number, dstOffset: number, size: number }} ReplayReadbackCopy */
/** @typedef {{ index: number, model: WordModel }} DynamicUniformWord */
/** @typedef {{ pos: number, staging: GPUBuffer, commandBuffer: GPUCommandBuffer }} EncodedToken */

/**
 * A method replaced on a WebGPU or recorder object during observation.
 * @typedef {{ target: object, name: string, own: boolean, previous: Function }} SavedPatch
 */

export const WHOLE_TOKEN_CAPTURE_STEPS = Object.freeze([2, 3, 4, 5]);

const U32 = 4;

/** @param {GPUBuffer | null | undefined} buffer */
function usageOf(buffer) {
  return typeof buffer?.usage === 'number' ? buffer.usage : 0;
}

/** @param {GPUBuffer | null | undefined} buffer */
function isUniform(buffer) {
  return (usageOf(buffer) & GPUBufferUsage.UNIFORM) !== 0;
}

/** @param {GPUBuffer | null | undefined} buffer */
function isMapRead(buffer) {
  return (usageOf(buffer) & GPUBufferUsage.MAP_READ) !== 0;
}

/**
 * @param {unknown} data
 * @param {number | undefined} dataOffset
 * @param {number | undefined} size
 * @returns {Uint8Array}
 */
function bytesOfWrite(data, dataOffset, size) {
  if (data instanceof ArrayBuffer || (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer)) {
    const start = dataOffset ?? 0;
    const length = size ?? (data.byteLength - start);
    return new Uint8Array(data, start, length).slice();
  }
  if (ArrayBuffer.isView(data)) {
    const elem = /** @type {ArrayBufferView & { BYTES_PER_ELEMENT?: number }} */ (data).BYTES_PER_ELEMENT ?? 1;
    const start = data.byteOffset + (dataOffset ?? 0) * elem;
    const count = size ?? ((data.byteLength / elem) - (dataOffset ?? 0));
    return new Uint8Array(data.buffer, start, count * elem).slice();
  }
  throw new Error('[WholeTokenProgram] Unsupported queue.writeBuffer data type.');
}

/**
 * @param {BindGroupEntryRecord} entry
 * @returns {{ buffer: GPUBuffer, offset: number, size: number | null } | null}
 */
function entryBuffer(entry) {
  const resource = /** @type {unknown} */ (entry.resource);
  if (resource && typeof resource === 'object' && 'buffer' in resource) {
    return {
      buffer: /** @type {GPUBufferBinding} */ (resource).buffer,
      offset: /** @type {GPUBufferBinding} */ (resource).offset ?? 0,
      size: /** @type {GPUBufferBinding} */ (resource).size ?? null,
    };
  }
  if (resource && typeof resource === 'object'
    && typeof /** @type {{ size?: unknown }} */ (resource).size === 'number'
    && typeof /** @type {{ usage?: unknown }} */ (resource).usage === 'number') {
    return { buffer: /** @type {GPUBuffer} */ (resource), offset: 0, size: null };
  }
  return null;
}

/**
 * One observed decode step. Owns nothing; records references only.
 */
class StepObservation {
  /**
   * @param {number} pos
   * @param {number} token
   */
  constructor(pos, token) {
    this.pos = pos;
    this.token = token;
    /** @type {ObservedOp[]} */
    this.ops = [];
    /** @type {Set<GPUBuffer>} */
    this.transient = new Set();
    /** @type {Map<GPUBuffer, Uint8Array>} */
    this.uniformBytes = new Map();
    /** @type {Set<string>} */
    this.writtenRegions = new Set();
    /** @type {string[]} */
    this.problems = [];
  }

  /** @param {string} message */
  problem(message) {
    if (this.problems.length < 8) this.problems.push(message);
  }

  /**
   * @param {GPUBuffer} buffer
   * @param {number} offset
   * @param {number | null} size
   * @returns {Uint8Array | null}
   */
  uniformSnapshot(buffer, offset, size) {
    const full = this.uniformBytes.get(buffer) ?? peekUniformCacheBytes(buffer);
    if (!full) {
      this.problem(`uniform buffer "${buffer.label ?? ''}" has no observable contents`);
      return null;
    }
    const length = size ?? (full.byteLength - offset);
    if (offset + length > full.byteLength) {
      this.problem(`uniform binding range exceeds known contents for "${buffer.label ?? ''}"`);
      return null;
    }
    return full.slice(offset, offset + length);
  }

  /**
   * @param {GPUBindGroup} bindGroup
   * @param {BindGroupDescriptors} descriptors
   * @returns {ObservedGroup}
   */
  snapshotBindGroup(bindGroup, descriptors) {
    const desc = descriptors.get(bindGroup);
    if (!desc) return { opaque: bindGroup };
    const entries = desc.entries.map((entry) => {
      const bound = entryBuffer(entry);
      if (!bound) {
        this.problem(`bind group "${desc.label ?? ''}" binds a non-buffer resource`);
        return { binding: entry.binding, other: entry.resource };
      }
      if (isUniform(bound.buffer)) {
        return {
          binding: entry.binding,
          uniform: this.uniformSnapshot(bound.buffer, bound.offset, bound.size),
        };
      }
      return { binding: entry.binding, buffer: bound.buffer, offset: bound.offset, size: bound.size };
    });
    return { layout: desc.layout, label: desc.label ?? null, entries };
  }
}

/**
 * @param {SavedPatch[]} saved
 * @param {object} target
 * @param {string} name
 * @param {(previous: Function) => Function} wrap
 */
function patchMethod(saved, target, name, wrap) {
  const own = Object.prototype.hasOwnProperty.call(target, name);
  const previous = /** @type {Record<string, unknown>} */ (target)[name];
  if (typeof previous !== 'function') throw new Error(`[WholeTokenProgram] Cannot observe missing method ${name}.`);
  saved.push({ target, name, own, previous });
  /** @type {Record<string, unknown>} */ (target)[name] = wrap(previous);
}

/** @param {SavedPatch[]} saved */
function restorePatches(saved) {
  for (let i = saved.length - 1; i >= 0; i--) {
    const { target, name, own, previous } = saved[i];
    if (own) /** @type {Record<string, unknown>} */ (target)[name] = previous;
    else delete /** @type {Record<string, unknown>} */ (target)[name];
  }
}

/**
 * @param {StepObservation} observation
 * @param {BindGroupDescriptors} descriptors
 * @param {GPUComputePassEncoder} pass
 * @returns {GPUComputePassEncoder}
 */
function wrapPass(observation, descriptors, pass) {
  /** @type {GPUComputePipeline | null} */
  let pipeline = null;
  /** @type {GPUBindGroup[]} */
  const groups = [];
  /** @type {ProxyHandler<GPUComputePassEncoder>} */
  const handler = {
    get(target, prop) {
      if (prop === 'setPipeline') return (/** @type {GPUComputePipeline} */ p) => { pipeline = p; target.setPipeline(p); };
      if (prop === 'setBindGroup') {
        return (
          /** @type {number} */ index,
          /** @type {GPUBindGroup} */ bindGroup,
          /** @type {(Iterable<number> & ArrayLike<number>) | undefined} */ offsets,
        ) => {
          if (offsets && offsets.length > 0) observation.problem('dynamic bind group offsets are not supported');
          groups[index] = bindGroup;
          return offsets === undefined ? target.setBindGroup(index, bindGroup) : target.setBindGroup(index, bindGroup, offsets);
        };
      }
      if (prop === 'dispatchWorkgroups') {
        return (/** @type {number} */ x, y = 1, z = 1) => {
          observation.ops.push({ kind: 'dispatch', pipeline, groups: groups.map((g) => observation.snapshotBindGroup(g, descriptors)), workgroups: [x, y, z] });
          target.dispatchWorkgroups(x, y, z);
        };
      }
      if (prop === 'dispatchWorkgroupsIndirect') {
        return (/** @type {GPUBuffer} */ buffer, offset = 0) => {
          observation.ops.push({ kind: 'indirect', pipeline, groups: groups.map((g) => observation.snapshotBindGroup(g, descriptors)), buffer, offset });
          target.dispatchWorkgroupsIndirect(buffer, offset);
        };
      }
      if (prop === 'end') return () => target.end();
      const value = /** @type {Record<string | symbol, unknown>} */ (/** @type {unknown} */ (target))[prop];
      if (typeof value === 'function') {
        return (/** @type {unknown[]} */ ...args) => { observation.problem(`compute pass method ${String(prop)} is not supported`); return value.apply(target, args); };
      }
      return value;
    },
  };
  return new Proxy(pass, handler);
}

/**
 * @param {StepObservation} observation
 * @param {BindGroupDescriptors} descriptors
 * @param {GPUCommandEncoder} encoder
 * @returns {GPUCommandEncoder}
 */
function wrapEncoder(observation, descriptors, encoder) {
  /** @type {ProxyHandler<GPUCommandEncoder>} */
  const handler = {
    get(target, prop) {
      if (prop === 'copyBufferToBuffer') {
        return (/** @type {[GPUBuffer, GPUBuffer | number, GPUBuffer | number, number?, number?]} */ ...args) => {
          const [src, a, b, c, d] = args;
          const op = /** @type {CopyOp} */ (args.length === 3
            ? { kind: 'copy', src, srcOffset: 0, dst: a, dstOffset: 0, size: b }
            : { kind: 'copy', src, srcOffset: a, dst: b, dstOffset: c, size: d ?? (src.size - /** @type {number} */ (a)) });
          observation.ops.push(op);
          return target.copyBufferToBuffer(.../** @type {[GPUBuffer, number, GPUBuffer, number, number?]} */ (args));
        };
      }
      if (prop === 'clearBuffer') {
        return (/** @type {GPUBuffer} */ buffer, offset = 0, /** @type {number | undefined} */ size) => {
          observation.ops.push({ kind: 'clear', buffer, offset, size: size ?? (buffer.size - offset) });
          return size === undefined ? target.clearBuffer(buffer, offset) : target.clearBuffer(buffer, offset, size);
        };
      }
      if (prop === 'beginComputePass') {
        return (/** @type {GPUComputePassDescriptor | undefined} */ desc) => wrapPass(observation, descriptors, target.beginComputePass(desc));
      }
      const value = /** @type {Record<string | symbol, unknown>} */ (/** @type {unknown} */ (target))[prop];
      if (typeof value === 'function') {
        return (/** @type {unknown[]} */ ...args) => { observation.problem(`command encoder method ${String(prop)} is not supported`); return value.apply(target, args); };
      }
      return value;
    },
  };
  return new Proxy(encoder, handler);
}

/**
 * Install observation hooks for one recorded decode step. Returns a stop function.
 * @param {GPUDevice} device
 * @param {CommandRecorder} recorder
 * @param {StepObservation} observation
 * @param {BindGroupDescriptors} descriptors
 * @returns {() => void}
 */
function observeStep(device, recorder, observation, descriptors) {
  /** @type {SavedPatch[]} */
  const saved = [];
  const queue = device.queue;
  let insideRecorderSubmit = 0;
  patchMethod(saved, queue, 'writeBuffer', (previous) => /**
     * @param {GPUBuffer} buffer
     * @param {number} offset
     * @param {unknown} data
     * @param {number | undefined} dataOffset
     * @param {number | undefined} size
     */ function writeBuffer(buffer, offset, data, dataOffset, size) {
    const bytes = bytesOfWrite(data, dataOffset, size);
    if (isUniform(buffer)) {
      const known = observation.uniformBytes.get(buffer) ?? new Uint8Array(buffer.size);
      const next = known.byteLength >= buffer.size ? known.slice() : new Uint8Array(buffer.size);
      next.set(bytes, offset);
      observation.uniformBytes.set(buffer, next);
    } else {
      observation.ops.push({ kind: 'write', buffer, offset, bytes });
    }
    return previous.apply(queue, arguments);
  });
  patchMethod(saved, queue, 'submit', (previous) => function submit() {
    if (insideRecorderSubmit === 0) observation.problem('the decode step submitted work outside its recorder');
    return previous.apply(queue, arguments);
  });
  for (const name of ['submit', 'submitPartial']) {
    patchMethod(saved, recorder, name, (previous) => function recorderSubmit() {
      insideRecorderSubmit++;
      try {
        return previous.apply(recorder, arguments);
      } finally {
        insideRecorderSubmit--;
      }
    });
  }
  patchMethod(saved, recorder, 'recordDispatch', (previous) => /**
     * @param {GPUComputePipeline} pipeline
     * @param {GPUBindGroup} bindGroup
     * @param {number[]} workgroups
     */ function recordDispatch(pipeline, bindGroup, workgroups) {
    observation.ops.push({
      kind: 'dispatch', pipeline, groups: [observation.snapshotBindGroup(bindGroup, descriptors)],
      workgroups: [workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1],
    });
    return previous.apply(recorder, arguments);
  });
  patchMethod(saved, recorder, 'recordDispatchIndirect', (previous) => /**
     * @param {GPUComputePipeline} pipeline
     * @param {GPUBindGroup} bindGroup
     * @param {GPUBuffer} buffer
     * @param {number} [offset]
     */ function recordDispatchIndirect(pipeline, bindGroup, buffer, offset = 0) {
    observation.ops.push({
      kind: 'indirect', pipeline, groups: [observation.snapshotBindGroup(bindGroup, descriptors)], buffer, offset,
    });
    return previous.apply(recorder, arguments);
  });
  patchMethod(saved, recorder, 'getEncoder', (previous) => function getEncoder() {
    return wrapEncoder(observation, descriptors, previous.call(recorder));
  });
  patchMethod(saved, recorder, 'beginComputePass', (previous) => function beginComputePass() {
    return wrapPass(observation, descriptors, previous.apply(recorder, arguments));
  });
  patchMethod(saved, recorder, 'createTempBuffer', (previous) => function createTempBuffer() {
    const buffer = previous.apply(recorder, arguments);
    observation.transient.add(buffer);
    return buffer;
  });
  patchMethod(saved, recorder, 'trackTemporaryBuffer', (previous) => /** @param {GPUBuffer} buffer */ function trackTemporaryBuffer(buffer) {
    if (!isPersistentBuffer(buffer) && isBufferActive(buffer)) observation.transient.add(buffer);
    return previous.call(recorder, buffer);
  });
  const stopRelease = observeBufferRelease((buffer) => observation.transient.add(buffer));
  return () => {
    stopRelease();
    restorePatches(saved);
  };
}

/**
 * Classify one u32 word across three observations.
 * @param {readonly number[]} values
 * @param {readonly number[]} positions
 * @param {readonly number[]} tokens
 * @returns {{ kind: 'const', value: number } | { kind: 'token' } | { kind: 'affine', base: number, slope: number, pos0: number } | null}
 */
export function classifyWord(values, positions, tokens) {
  if (values.every((v) => v === values[0])) return { kind: 'const', value: values[0] };
  if (values.every((v, i) => v === (tokens[i] >>> 0))) return { kind: 'token' };
  for (let i = 1; i < positions.length; i++) if (positions[i] - positions[i - 1] !== 1) return null;
  const slope = (values[1] - values[0]) | 0;
  for (let i = 2; i < values.length; i++) if (((values[i] - values[i - 1]) | 0) !== slope) return null;
  return { kind: 'affine', base: values[0], slope, pos0: positions[0] };
}

/**
 * @param {WordModel} model
 * @param {number} pos
 * @param {number} token
 * @returns {number}
 */
function evaluateWord(model, pos, token) {
  if (model.kind === 'const') return model.value;
  if (model.kind === 'token') return token >>> 0;
  return (model.base + model.slope * (pos - model.pos0)) >>> 0;
}

/**
 * Compare three byte arrays word by word. Returns dynamic word models or throws.
 * @param {Uint8Array[]} arrays
 * @param {StepObservation[]} observations
 * @param {string} where
 * @returns {DynamicWord[]}
 */
function classifyBytes(arrays, observations, where) {
  const first = arrays[0];
  if (arrays.some((a) => a.byteLength !== first.byteLength)) {
    throw new Error(`[WholeTokenProgram] ${where}: byte length changes between decode steps.`);
  }
  if (first.byteLength % U32 !== 0) {
    if (!arrays.every((a) => a.every((v, i) => v === first[i]))) {
      throw new Error(`[WholeTokenProgram] ${where}: unaligned contents change between decode steps.`);
    }
    return [];
  }
  const words = arrays.map((a) => new Uint32Array(a.buffer, a.byteOffset, a.byteLength / U32));
  const positions = observations.map((o) => o.pos);
  const tokens = observations.map((o) => o.token);
  /** @type {DynamicWord[]} */
  const dynamic = [];
  for (let i = 0; i < words[0].length; i++) {
    const values = words.map((w) => w[i]);
    if (values.every((v) => v === values[0])) continue;
    const model = classifyWord(values, positions, tokens);
    if (!model) {
      throw new Error(
        `[WholeTokenProgram] ${where}: word ${i} (${values.join(', ')}) is neither the token id nor affine in the position.`
      );
    }
    dynamic.push({ word: i, model });
  }
  return dynamic;
}

/**
 * @param {number} value
 * @param {number} alignment
 */
function alignUp(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

export class WholeTokenProgram {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    /** @type {BindGroupDescriptors} */
    this.descriptors = new WeakMap();
    // Bind group descriptors are observed from the generation's first decode step,
    // so groups a kernel caches in step 1 and reuses later stay inspectable.
    /** @type {SavedPatch[]} */
    this.watch = [];
    patchMethod(this.watch, device, 'createBindGroup', (previous) => (/** @type {GPUBindGroupDescriptor} */ desc) => {
      const bindGroup = previous.call(device, desc);
      this.descriptors.set(bindGroup, {
        layout: desc.layout,
        label: desc.label ?? null,
        entries: Array.from(desc.entries, (entry) => ({ binding: entry.binding, resource: entry.resource })),
      });
      return bindGroup;
    });
    /** @type {StepObservation[]} */
    this.observations = [];
    this.ready = false;
    /** @type {number | null} */
    this.nextPos = null;
    /** @type {GPUBuffer[]} */
    this.owned = [];
    /** @type {EncodedToken | null} */
    this.pending = null;
    this.stagingIndex = 0;
    this.stats = { observedSteps: 0, replayedTokens: 0 };
    // Built by #materialize once the observed steps are accepted (declarations only).
    /** @type {ReplayStep[]} */ this.steps;
    /** @type {ReplayWrite[]} */ this.writes;
    /** @type {ReplayReadbackCopy[]} */ this.readback;
    /** @type {number} */ this.readbackBytes;
    /** @type {GPUBuffer[]} */ this.staging;
    /** @type {number} */ this.dispatchCount;
    /** @type {GPUBuffer} */ this.uniformBuffer;
    /** @type {ArrayBuffer} */ this.uniformMirror;
    /** @type {Uint32Array} */ this.uniformWords;
    /** @type {number} */ this.dynamicUniformBytes;
    /** @type {DynamicUniformWord[]} */ this.dynamicUniformWords;
    /** @type {number | undefined} */ this.positionDependentCopies;
  }

  /**
   * Observe one ordinary recorded decode step. `run` executes it and returns the token.
   * @template T
   * @param {number} pos
   * @param {number} token
   * @param {(attach: (recorder: CommandRecorder) => void) => Promise<T>} run
   * @returns {Promise<T>}
   */
  async observe(pos, token, run) {
    if (this.ready) throw new Error('[WholeTokenProgram] Program is already built.');
    const observation = new StepObservation(pos, token);
    /** @type {(() => void) | null} */
    let stop = null;
    const attach = (/** @type {CommandRecorder} */ recorder) => {
      if (stop) throw new Error('[WholeTokenProgram] A decode step created more than one recorder.');
      stop = observeStep(this.device, recorder, observation, this.descriptors);
    };
    let result;
    try {
      result = await run(attach);
    } catch (error) {
      this.#unwatch();
      throw error;
    } finally {
      /** @type {(() => void) | null} */ (stop)?.();
    }
    if (!stop) throw new Error('[WholeTokenProgram] The observed decode step created no recorder.');
    if (observation.problems.length > 0) {
      throw new Error(`[WholeTokenProgram] Decode step at position ${pos} cannot become a token program: ${observation.problems.join('; ')}.`);
    }
    this.observations.push(observation);
    this.stats.observedSteps++;
    if (this.observations.length === WHOLE_TOKEN_CAPTURE_STEPS.length) {
      this.#unwatch();
      this.#build();
      this.observations = [];
      this.nextPos = pos + 1;
    }
    return result;
  }

  /**
   * @param {StepObservation} observation
   * @returns {(buffer: GPUBuffer) => BufferKey}
   */
  #canonicalBuffers(observation) {
    /** @type {Map<GPUBuffer, string>} */
    const ids = new Map();
    return (buffer) => {
      if (!observation.transient.has(buffer)) return buffer;
      let id = ids.get(buffer);
      if (id === undefined) {
        id = `transient:${ids.size}`;
        ids.set(buffer, id);
      }
      return id;
    };
  }

  #build() {
    const obs = this.observations;
    const lengths = obs.map((o) => o.ops.length);
    if (lengths.some((n) => n !== lengths[0])) {
      throw new Error(`[WholeTokenProgram] Decode steps recorded ${lengths.join('/')} commands; the structure changes with position.`);
    }
    const canon = obs.map((o) => this.#canonicalBuffers(o));
    /** @type {Map<string, TransientInfo>} */
    const transientTemplate = new Map();
    /**
     * @param {BufferKey} key
     * @param {GPUBuffer} buffer
     */
    const noteTransient = (key, buffer) => {
      if (typeof key !== 'string') return;
      const known = transientTemplate.get(key);
      if (!known) transientTemplate.set(key, { size: buffer.size, usage: buffer.usage });
      else if (known.size !== buffer.size || known.usage !== buffer.usage) {
        throw new Error(`[WholeTokenProgram] Transient ${key} changes size or usage between decode steps.`);
      }
    };
    /**
     * @param {number} index
     * @param {string} field
     * @param {GPUBuffer[]} buffers
     * @returns {BufferKey}
     */
    const sameBuffer = (index, field, buffers) => {
      const keys = buffers.map((buffer, i) => canon[i](buffer));
      if (keys.some((key) => key !== keys[0])) {
        throw new Error(`[WholeTokenProgram] Command ${index} binds a different ${field} buffer in each decode step.`);
      }
      buffers.forEach((buffer) => noteTransient(keys[0], buffer));
      return keys[0];
    };

    const minAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    /** @type {UniformSlot[]} */
    const uniformSlots = [];
    /** @type {TemplateStep[]} */
    const template = [];
    /** @type {ReadbackTemplate | null} */
    let readback = null;
    for (let i = 0; i < lengths[0]; i++) {
      const ops = /** @type {ObservedOpView[]} */ (/** @type {unknown} */ (obs.map((o) => o.ops[i])));
      const kind = ops[0].kind;
      if (ops.some((op) => op.kind !== kind)) {
        throw new Error(`[WholeTokenProgram] Command ${i} changes kind between decode steps.`);
      }
      if (kind === 'dispatch' || kind === 'indirect') {
        if (ops.some((op) => op.pipeline !== ops[0].pipeline)) {
          throw new Error(`[WholeTokenProgram] Command ${i} uses a different pipeline in each decode step.`);
        }
        /** @type {DispatchTemplate} */
        const step = { kind, pipeline: ops[0].pipeline, groups: [] };
        if (kind === 'dispatch') {
          const w = ops.map((op) => op.workgroups.join(','));
          if (w.some((g) => g !== w[0])) {
            throw new Error(`[WholeTokenProgram] Command ${i} dispatch geometry changes with position (${w.join(' | ')}).`);
          }
          step.workgroups = ops[0].workgroups;
        } else {
          step.indirect = sameBuffer(i, 'indirect', ops.map((op) => op.buffer));
          if (ops.some((op) => op.offset !== ops[0].offset)) throw new Error(`[WholeTokenProgram] Command ${i} indirect offset changes.`);
          step.indirectOffset = ops[0].offset;
        }
        const groupCount = ops[0].groups.length;
        if (ops.some((op) => op.groups.length !== groupCount)) throw new Error(`[WholeTokenProgram] Command ${i} bind group count changes.`);
        for (let g = 0; g < groupCount; g++) {
          const groups = ops.map((op) => op.groups[g]);
          if (!groups[0]) {
            if (groups.some(Boolean)) throw new Error(`[WholeTokenProgram] Command ${i} group ${g} presence changes.`);
            step.groups.push(null);
            continue;
          }
          if (groups[0].opaque) {
            if (groups.some((group) => group.opaque !== groups[0].opaque)) {
              throw new Error(`[WholeTokenProgram] Command ${i} group ${g} is a bind group created outside the observed steps that changes between them.`);
            }
            step.groups.push({ opaque: groups[0].opaque });
            continue;
          }
          if (groups.some((group) => group.opaque || group.layout !== groups[0].layout || group.entries.length !== groups[0].entries.length)) {
            throw new Error(`[WholeTokenProgram] Command ${i} group ${g} layout changes between decode steps.`);
          }
          /** @type {TemplateEntry[]} */
          const entries = [];
          for (let e = 0; e < groups[0].entries.length; e++) {
            const items = groups.map((group) => group.entries[e]);
            if (items.some((item) => item.binding !== items[0].binding)) throw new Error(`[WholeTokenProgram] Command ${i} binding order changes.`);
            if (items[0].other !== undefined) throw new Error(`[WholeTokenProgram] Command ${i} binds a non-buffer resource.`);
            if (items[0].uniform !== undefined) {
              if (items.some((item) => !item.uniform)) throw new Error(`[WholeTokenProgram] Command ${i} uniform binding ${items[0].binding} has unknown contents.`);
              const where = `command ${i} (${groups[0].label ?? 'bind group'}) uniform binding ${items[0].binding}`;
              const dynamic = classifyBytes(/** @type {Uint8Array[]} */ (items.map((item) => item.uniform)), obs, where);
              const slot = { bytes: /** @type {Uint8Array} */ (items[0].uniform), dynamic, offset: 0 };
              uniformSlots.push(slot);
              entries.push({ binding: items[0].binding, slot });
              continue;
            }
            const key = sameBuffer(i, `binding ${items[0].binding}`, items.map((item) => item.buffer));
            if (items.some((item) => item.offset !== items[0].offset || item.size !== items[0].size)) {
              throw new Error(`[WholeTokenProgram] Command ${i} binding ${items[0].binding} range changes.`);
            }
            entries.push({ binding: items[0].binding, key, offset: items[0].offset, size: items[0].size });
          }
          step.groups.push({ layout: groups[0].layout, label: groups[0].label, entries });
        }
        template.push(step);
        continue;
      }
      if (kind === 'write') {
        const key = sameBuffer(i, 'write target', ops.map((op) => op.buffer));
        if (ops.some((op) => op.offset !== ops[0].offset)) throw new Error(`[WholeTokenProgram] Command ${i} write offset changes.`);
        const dynamic = classifyBytes(ops.map((op) => op.bytes), obs, `command ${i} queue write`);
        template.push({ kind, key, offset: ops[0].offset, bytes: ops[0].bytes, dynamic });
        continue;
      }
      if (kind === 'copy') {
        if (ops.some((op) => op.size !== ops[0].size)) {
          throw new Error(`[WholeTokenProgram] Command ${i} copy size changes.`);
        }
        // Offsets may advance with the position (KV appends); the size may not.
        /**
         * @param {'srcOffset' | 'dstOffset'} field
         * @returns {WordModel}
         */
        const offsetModel = (field) => {
          const model = classifyWord(ops.map((op) => op[field]), obs.map((o) => o.pos), obs.map((o) => o.token));
          if (!model || model.kind === 'token') {
            throw new Error(`[WholeTokenProgram] Command ${i} copy ${field} is neither constant nor affine in the position.`);
          }
          return model;
        };
        const srcOffsetModel = offsetModel('srcOffset');
        const dstOffsetModel = offsetModel('dstOffset');
        const src = sameBuffer(i, 'copy source', ops.map((op) => op.src));
        if (isMapRead(ops[0].dst)) {
          if (srcOffsetModel.kind !== 'const' || dstOffsetModel.kind !== 'const') {
            throw new Error('[WholeTokenProgram] Readback copy offsets must not depend on the position.');
          }
          // The step ends with copies into its mappable staging buffer: the sampled token
          // at offset 0, optionally followed by the finiteness guard words.
          readback ??= { copies: [], dst: ops.map((op) => op.dst) };
          if (ops.some((op, n) => op.dst !== /** @type {ReadbackTemplate} */ (readback).dst[n])) {
            throw new Error('[WholeTokenProgram] Decode step copies to more than one mappable buffer.');
          }
          if (ops[0].size % U32 !== 0 || ops[0].dstOffset % U32 !== 0) {
            throw new Error('[WholeTokenProgram] Readback copies must be whole words.');
          }
          readback.copies.push({ src, srcOffset: ops[0].srcOffset, dstOffset: ops[0].dstOffset, size: ops[0].size });
          continue;
        }
        if (readback) throw new Error('[WholeTokenProgram] Decode step records commands after its readback copies.');
        const dst = sameBuffer(i, 'copy destination', ops.map((op) => op.dst));
        template.push({ kind, src, srcOffset: srcOffsetModel, dst, dstOffset: dstOffsetModel, size: ops[0].size });
        continue;
      }
      if (kind === 'clear') {
        const key = sameBuffer(i, 'clear target', ops.map((op) => op.buffer));
        if (ops.some((op) => op.offset !== ops[0].offset || op.size !== ops[0].size)) throw new Error(`[WholeTokenProgram] Command ${i} clear range changes.`);
        template.push({ kind, key, offset: ops[0].offset, size: ops[0].size });
        continue;
      }
      throw new Error(`[WholeTokenProgram] Unknown command kind ${kind}.`);
    }
    if (!readback || !readback.copies.some((c) => c.dstOffset === 0 && c.size === U32)) {
      throw new Error('[WholeTokenProgram] Decode step has no 4-byte sampled-token readback at offset 0.');
    }

    // Writes: one write per region, else the single-submit program would reorder them.
    /** @type {Set<{ key: BufferKey, offset: number }>} */
    const written = new Set();
    for (const step of template) {
      if (step.kind !== 'write') continue;
      const region = `${typeof step.key === 'string' ? step.key : step.key.label}:${step.offset}:${step.bytes.byteLength}`;
      const id = { key: step.key, offset: step.offset };
      for (const prior of written) {
        if (prior.key === id.key && prior.offset === id.offset) {
          throw new Error(`[WholeTokenProgram] Decode step writes ${region} more than once.`);
        }
      }
      written.add(id);
    }

    this.#materialize(template, uniformSlots, transientTemplate, readback, minAlign);
  }

  /**
   * @param {GPUBufferDescriptor} descriptor
   * @returns {GPUBuffer}
   */
  #createOwnedBuffer(descriptor) {
    const buffer = this.device.createBuffer(descriptor);
    this.owned.push(buffer);
    return buffer;
  }

  /**
   * @param {TemplateStep[]} template
   * @param {UniformSlot[]} uniformSlots
   * @param {Map<string, TransientInfo>} transientTemplate
   * @param {ReadbackTemplate} readback
   * @param {number} minAlign
   */
  #materialize(template, uniformSlots, transientTemplate, readback, minAlign) {
    const device = this.device;
    /** @type {Map<string, GPUBuffer>} */
    const substitutes = new Map();
    for (const [key, info] of transientTemplate) {
      substitutes.set(key, this.#createOwnedBuffer({ label: `whole_token_${key}`, size: info.size, usage: info.usage }));
    }
    const resolve = (/** @type {BufferKey} */ key) => /** @type {GPUBuffer} */ (typeof key === 'string' ? substitutes.get(key) : key);

    // Dynamic uniform slots first so one prefix write per token updates them all.
    const dynamicSlots = uniformSlots.filter((slot) => slot.dynamic.length > 0);
    const constantSlots = uniformSlots.filter((slot) => slot.dynamic.length === 0);
    let offset = 0;
    for (const slot of [...dynamicSlots, ...constantSlots]) {
      slot.offset = offset;
      offset = alignUp(offset + slot.bytes.byteLength, minAlign);
    }
    this.dynamicUniformBytes = dynamicSlots.length > 0
      ? alignUp(dynamicSlots[dynamicSlots.length - 1].offset + dynamicSlots[dynamicSlots.length - 1].bytes.byteLength, U32)
      : 0;
    const packedSize = Math.max(alignUp(offset, 16), 16);
    this.uniformMirror = new ArrayBuffer(packedSize);
    const mirrorBytes = new Uint8Array(this.uniformMirror);
    for (const slot of uniformSlots) mirrorBytes.set(slot.bytes, slot.offset);
    this.uniformWords = new Uint32Array(this.uniformMirror);
    this.uniformBuffer = this.#createOwnedBuffer({
      label: 'whole_token_uniforms', size: packedSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformMirror);
    this.dynamicUniformWords = [];
    for (const slot of dynamicSlots) {
      for (const { word, model } of slot.dynamic) this.dynamicUniformWords.push({ index: slot.offset / U32 + word, model });
    }

    this.writes = [];
    this.steps = [];
    for (const step of template) {
      if (step.kind === 'write') {
        this.writes.push({ buffer: resolve(step.key), offset: step.offset, bytes: step.bytes.slice(), dynamic: step.dynamic });
        continue;
      }
      if (step.kind === 'copy') {
        this.steps.push({ kind: 'copy', src: resolve(step.src), srcOffset: step.srcOffset, dst: resolve(step.dst), dstOffset: step.dstOffset, size: step.size });
        this.positionDependentCopies = (this.positionDependentCopies ?? 0)
          + (step.srcOffset.kind !== 'const' || step.dstOffset.kind !== 'const' ? 1 : 0);
        continue;
      }
      if (step.kind === 'clear') {
        this.steps.push({ kind: 'clear', buffer: resolve(step.key), offset: step.offset, size: step.size });
        continue;
      }
      const groups = step.groups.map((group) => {
        if (!group) return null;
        if (group.opaque) return group.opaque;
        return device.createBindGroup({
          label: group.label ? `whole_token_${group.label}` : 'whole_token_bind_group',
          layout: group.layout,
          entries: group.entries.map((entry) => {
            if (entry.slot) {
              return { binding: entry.binding, resource: { buffer: this.uniformBuffer, offset: entry.slot.offset, size: entry.slot.bytes.byteLength } };
            }
            /** @type {GPUBufferBinding} */
            const resource = { buffer: resolve(entry.key), offset: entry.offset };
            if (entry.size !== null && entry.size !== undefined) resource.size = entry.size;
            return { binding: entry.binding, resource };
          }),
        });
      });
      this.steps.push(step.kind === 'dispatch'
        ? { kind: 'dispatch', pipeline: step.pipeline, groups, workgroups: /** @type {number[]} */ (step.workgroups) }
        : { kind: 'indirect', pipeline: step.pipeline, groups, indirect: resolve(/** @type {BufferKey} */ (step.indirect)), indirectOffset: /** @type {number} */ (step.indirectOffset) });
    }
    this.readback = readback.copies.map((c) => ({ ...c, src: resolve(c.src) }));
    this.readbackBytes = Math.max(...this.readback.map((c) => c.dstOffset + c.size));
    this.staging = [0, 1].map((i) => this.#createOwnedBuffer({
      label: `whole_token_readback_${i}`, size: alignUp(this.readbackBytes, U32), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }));
    this.dispatchCount = this.steps.filter((s) => s.kind !== 'copy' && s.kind !== 'clear').length;
    this.ready = true;
    log.info('Decode', `Whole-token program: ${this.dispatchCount} dispatches, ${uniformSlots.length} uniform slots `
      + `(${dynamicSlots.length} position/token dependent), ${substitutes.size} private transients, ${this.writes.length} queue writes, `
      + `${this.positionDependentCopies ?? 0} position-dependent copies.`);
  }

  /**
   * @param {number} pos
   * @param {number} stagingIndex
   * @returns {EncodedToken}
   */
  #encode(pos, stagingIndex) {
    const encoder = this.device.createCommandEncoder({ label: `whole_token_${pos}` });
    /** @type {GPUComputePassEncoder | null} */
    let pass = null;
    /** @type {GPUComputePipeline | null} */
    let lastPipeline = null;
    for (const step of this.steps) {
      if (step.kind === 'copy' || step.kind === 'clear') {
        if (pass) { pass.end(); pass = null; lastPipeline = null; }
        if (step.kind === 'copy') {
          encoder.copyBufferToBuffer(step.src, evaluateWord(step.srcOffset, pos, 0), step.dst,
            evaluateWord(step.dstOffset, pos, 0), step.size);
        }
        else encoder.clearBuffer(step.buffer, step.offset, step.size);
        continue;
      }
      if (!pass) pass = encoder.beginComputePass({ label: 'whole_token_pass' });
      if (step.pipeline !== lastPipeline) { pass.setPipeline(step.pipeline); lastPipeline = step.pipeline; }
      for (let g = 0; g < step.groups.length; g++) if (step.groups[g]) pass.setBindGroup(g, step.groups[g]);
      if (step.kind === 'dispatch') pass.dispatchWorkgroups(step.workgroups[0], step.workgroups[1], step.workgroups[2]);
      else pass.dispatchWorkgroupsIndirect(step.indirect, step.indirectOffset);
    }
    if (pass) pass.end();
    const staging = this.staging[stagingIndex];
    for (const c of this.readback) encoder.copyBufferToBuffer(c.src, c.srcOffset, staging, c.dstOffset, c.size);
    return { pos, staging, commandBuffer: encoder.finish() };
  }

  /**
   * @param {number} pos
   * @param {number} token
   */
  #writeInputs(pos, token) {
    const queue = this.device.queue;
    if (this.dynamicUniformWords.length > 0) {
      for (const { index, model } of this.dynamicUniformWords) this.uniformWords[index] = evaluateWord(model, pos, token);
      queue.writeBuffer(this.uniformBuffer, 0, this.uniformMirror, 0, this.dynamicUniformBytes);
    }
    for (const write of this.writes) {
      if (write.dynamic.length > 0) {
        const words = new Uint32Array(write.bytes.buffer, write.bytes.byteOffset, write.bytes.byteLength / U32);
        for (const { word, model } of write.dynamic) words[word] = evaluateWord(model, pos, token);
      }
      queue.writeBuffer(write.buffer, write.offset, write.bytes);
    }
  }

  /**
   * Run one token at `pos` whose input token id is `token`. Returns the readback words:
   * word 0 is the sampled token id, later words are the step's other readback copies.
   * @param {number} pos
   * @param {number} token
   * @returns {Promise<Uint32Array>}
   */
  async run(pos, token) {
    if (!this.ready) throw new Error('[WholeTokenProgram] Program is not built.');
    if (pos !== this.nextPos) throw new Error(`[WholeTokenProgram] Expected position ${this.nextPos}, got ${pos}.`);
    const pending = this.pending && this.pending.pos === pos ? this.pending : null;
    this.pending = null;
    const job = pending ?? this.#encode(pos, this.stagingIndex);
    this.#writeInputs(pos, token);
    this.device.queue.submit([job.commandBuffer]);
    const mapped = job.staging.mapAsync(GPUMapMode.READ, 0, this.staging[0].size);
    this.stagingIndex = this.staging.indexOf(job.staging) ^ 1;
    this.pending = this.#encode(pos + 1, this.stagingIndex);
    await mapped;
    const words = new Uint32Array(job.staging.getMappedRange(0, this.staging[0].size)).slice();
    job.staging.unmap();
    this.nextPos = pos + 1;
    this.stats.replayedTokens++;
    return words;
  }

  #unwatch() {
    restorePatches(this.watch);
    this.watch = [];
  }

  destroy() {
    this.#unwatch();
    this.pending = null;
    for (const buffer of this.owned) buffer.destroy();
    this.owned = [];
    this.ready = false;
  }
}
