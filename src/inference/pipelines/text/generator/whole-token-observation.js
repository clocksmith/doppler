import { peekUniformCacheBytes } from '../../../../gpu/uniform-cache.js';
import { observeStepContract } from '../../../../gpu/step-contract.js';
import { getBufferPool, isBufferActive, isPersistentBuffer } from '../../../../memory/buffer-pool.js';

// Observation of one ordinary recorded decode step for the whole-token program
// (whole-token-program.js): every GPU command the step records (dispatches, bind
// group descriptors, uniform contents, queue writes, copies and clears), the
// buffers that live only for the step, and the step contract its operations
// declare on the recorder (declareStepContract): the queue write that carries the
// input token id, and the step positions over which their commands stay fixed or
// affine in the position.

/** @typedef {import('../../../../gpu/command-recorder.js').CommandRecorder} CommandRecorder */

/**
 * A u32 word as a function of the decode position and input token.
 * @typedef {{ kind: 'const', value: number } | { kind: 'token' } | { kind: 'affine', base: number, slope: number, pos0: number }} WordModel
 */

/** @typedef {{ word: number, model: WordModel }} DynamicWord */
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
/** @typedef {{ kind: 'write', buffer: GPUBuffer, offset: number, bytes: Uint8Array, token: boolean }} WriteOp */
/** @typedef {{ kind: 'copy', src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number }} CopyOp */
/** @typedef {{ kind: 'clear', buffer: GPUBuffer, offset: number, size: number }} ClearOp */
/** @typedef {DispatchOp | IndirectOp | WriteOp | CopyOp | ClearOp} ObservedOp */
/** @typedef {import('../../../../gpu/step-contract.js').StepContractDeclaration} StepContractDeclaration */
/** A declared range of step positions (inclusive) and the operation that declared it. @typedef {{ owner: string, min: number, max: number }} PositionLimit */
/** @typedef {{ uniform: Uint8Array, known: Uint8Array }} UniformContents */

/**
 * A method replaced on a WebGPU or recorder object during observation.
 * @typedef {{ target: object, name: string, own: boolean, previous: Function }} SavedPatch
 */
export const U32 = 4;

/** @param {GPUBuffer | null | undefined} buffer */
function usageOf(buffer) {
  return typeof buffer?.usage === 'number' ? buffer.usage : 0;
}

/** @param {GPUBuffer | null | undefined} buffer */
function isUniform(buffer) {
  return (usageOf(buffer) & GPUBufferUsage.UNIFORM) !== 0;
}

/** @param {GPUBuffer | null | undefined} buffer */
export function isMapRead(buffer) {
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
export class StepObservation {
  /**
   * @param {number} pos
   * @param {number} token
   */
  constructor(pos, token) {
    this.pos = pos;
    this.token = token;
    /** Borrowed GPU token input for an explicitly declared variable-advance window. @type {GPUBuffer | null} */
    this.gpuTokenInput = null;
    /** @type {ObservedOp[]} */
    this.ops = [];
    /** @type {Set<GPUBuffer>} */
    this.transient = new Set();
    /** Uniform buffers written during the step: contents and which bytes are known. @type {Map<GPUBuffer, UniformContents>} */
    this.uniformBytes = new Map();
    /** Uniform buffers whose contents an observed bind group entry reads. @type {Set<GPUBuffer>} */
    this.boundUniforms = new Set();
    /** GPU-written uniforms retain buffer identity instead of becoming packed constants. @type {Set<GPUBuffer>} */
    this.gpuWrittenUniforms = new Set();
    /** Declared token inputs not yet matched to their queue write. @type {{ buffer: GPUBuffer, offset: number }[]} */
    this.pendingTokenInputs = [];
    /** @type {PositionLimit[]} */
    this.limits = [];
    /** Intersection of `limits`, set when the step finishes. @type {PositionLimit | null} */
    this.rangeLimit = null;
    /** Resources the caller bound the step to, read after the step ran. @type {readonly unknown[]} */
    this.identity = [];
    /** @type {string[]} */
    this.problems = [];
  }

  /** @param {string} message */
  problem(message) {
    if (this.problems.length < 8) this.problems.push(message);
  }

  /** @param {StepContractDeclaration} declaration */
  declare(declaration) {
    if (declaration.kind === 'token-input') {
      if (declaration.count !== 1 || !Number.isInteger(declaration.offset) || declaration.offset % U32 !== 0) {
        this.problem(`a token input must be one aligned u32 word (count ${declaration.count}, offset ${declaration.offset})`);
        return;
      }
      this.pendingTokenInputs.push({ buffer: declaration.buffer, offset: declaration.offset });
      return;
    }
    if (declaration.kind === 'position-range') {
      const { owner, min, max } = declaration;
      if (typeof min !== 'number' || typeof max !== 'number' || Number.isNaN(min) || Number.isNaN(max) || min > max) {
        this.problem(`${owner} declares an invalid position range [${min}, ${max}]`);
        return;
      }
      this.limits.push({ owner, min, max });
      return;
    }
    if (declaration.kind === 'context-limit') {
      const { owner, value, max } = declaration;
      if (!Number.isFinite(value) || !Number.isFinite(max) || value > max) {
        this.problem(`${owner} declares an invalid context limit (value ${value}, max ${max})`);
        return;
      }
      this.limits.push({ owner, min: -Infinity, max: this.pos + (max - value) });
      return;
    }
    this.problem(`unknown step-contract declaration "${String(/** @type {{ kind?: unknown }} */ (declaration).kind)}"`);
  }

  /**
   * Whether a queue write is a declared token input (and consume the declaration).
   * @param {GPUBuffer} buffer
   * @param {number} offset
   * @param {Uint8Array} bytes
   */
  takeTokenInput(buffer, offset, bytes) {
    const index = this.pendingTokenInputs.findIndex((input) => input.buffer === buffer && input.offset === offset);
    if (index < 0) return false;
    this.pendingTokenInputs.splice(index, 1);
    const word = bytes.byteLength === U32 ? new DataView(bytes.buffer, bytes.byteOffset, U32).getUint32(0, true) : null;
    if (word !== (this.token >>> 0)) {
      this.problem(`the declared token input of "${buffer.label ?? ''}" does not hold the step's input token ${this.token}`);
    }
    return true;
  }

  /**
   * @param {GPUBuffer} buffer
   * @param {number} offset
   * @param {Uint8Array} bytes
   */
  writeUniform(buffer, offset, bytes) {
    let contents = this.uniformBytes.get(buffer);
    if (!contents) {
      contents = { uniform: new Uint8Array(buffer.size), known: new Uint8Array(buffer.size) };
      this.uniformBytes.set(buffer, contents);
    }
    contents.uniform.set(bytes, offset);
    contents.known.fill(1, offset, offset + bytes.byteLength);
  }

  /**
   * Intersection of the declared position ranges; must contain this step's position.
   * @returns {PositionLimit}
   */
  range() {
    /** @type {PositionLimit} */
    let range = { owner: 'none', min: -Infinity, max: Infinity };
    for (const limit of this.limits) {
      range = {
        owner: limit.max < range.max ? limit.owner : range.owner,
        min: Math.max(range.min, limit.min),
        max: Math.min(range.max, limit.max),
      };
      if (this.pos < limit.min || this.pos > limit.max) {
        this.problem(`${limit.owner} declares positions [${limit.min}, ${limit.max}], which exclude its own step at ${this.pos}`);
      }
    }
    return range;
  }

  /**
   * @param {GPUBuffer} buffer
   * @param {number} offset
   * @param {number | null} size
   * @returns {Uint8Array | null}
   */
  uniformSnapshot(buffer, offset, size) {
    const written = this.uniformBytes.get(buffer);
    const full = written?.uniform ?? peekUniformCacheBytes(buffer);
    if (!full) {
      this.problem(`uniform buffer "${buffer.label ?? ''}" has no observable contents`);
      return null;
    }
    const length = size ?? (full.byteLength - offset);
    if (offset + length > full.byteLength) {
      this.problem(`uniform binding range exceeds known contents for "${buffer.label ?? ''}"`);
      return null;
    }
    // A partial write leaves earlier, unobserved bytes in the binding range.
    if (written && !written.known.subarray(offset, offset + length).every((flag) => flag === 1)) {
      this.problem(`uniform binding of "${buffer.label ?? ''}" reads bytes the step did not write`);
      return null;
    }
    this.boundUniforms.add(buffer);
    return full.slice(offset, offset + length);
  }

  /** @param {GPUBuffer} buffer */
  preserveGpuWrittenUniform(buffer) {
    if (!isUniform(buffer) || this.gpuWrittenUniforms.has(buffer)) return;
    const bytes = this.uniformSnapshot(buffer, 0, buffer.size);
    if (bytes) this.ops.push({ kind: 'write', buffer, offset: 0, bytes, token: false });
    this.gpuWrittenUniforms.add(buffer);
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
      if (isUniform(bound.buffer) && !this.gpuWrittenUniforms.has(bound.buffer)) {
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
export function patchMethod(saved, target, name, wrap) {
  const own = Object.prototype.hasOwnProperty.call(target, name);
  const previous = /** @type {Record<string, unknown>} */ (target)[name];
  if (typeof previous !== 'function') throw new Error(`[WholeTokenProgram] Cannot observe missing method ${name}.`);
  saved.push({ target, name, own, previous });
  /** @type {Record<string, unknown>} */ (target)[name] = wrap(previous);
}

/** @param {SavedPatch[]} saved */
export function restorePatches(saved) {
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
          observation.preserveGpuWrittenUniform(op.dst);
          observation.ops.push(op);
          return target.copyBufferToBuffer(.../** @type {[GPUBuffer, number, GPUBuffer, number, number?]} */ (args));
        };
      }
      if (prop === 'clearBuffer') {
        return (/** @type {GPUBuffer} */ buffer, offset = 0, /** @type {number | undefined} */ size) => {
          observation.preserveGpuWrittenUniform(buffer);
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
export function observeStep(device, recorder, observation, descriptors) {
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
    const token = observation.takeTokenInput(buffer, offset, bytes);
    if (isUniform(buffer)) {
      if (token) observation.problem(`the declared token input targets uniform buffer "${buffer.label ?? ''}"`);
      observation.writeUniform(buffer, offset, bytes);
    } else {
      observation.ops.push({ kind: 'write', buffer, offset, bytes, token });
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
  // Buffers returned to the pool during the step are its transients too.
  const pool = getBufferPool(device);
  for (const name of ['release', 'discard']) {
    patchMethod(saved, pool, name, (previous) => /** @param {GPUBuffer} buffer */ function released(buffer) {
      if (isBufferActive(buffer)) observation.transient.add(buffer);
      return previous.apply(pool, arguments);
    });
  }
  const stopContract = observeStepContract(recorder, (declaration) => observation.declare(declaration));
  return () => {
    stopContract();
    restorePatches(saved);
  };
}
