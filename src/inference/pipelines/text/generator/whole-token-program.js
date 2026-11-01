import { isBufferActive } from '../../../../memory/buffer-pool.js';
import { log } from '../../../../debug/index.js';
import { StepObservation, observeStep, patchMethod, restorePatches, U32 } from './whole-token-observation.js';
import { affineLastPosition, buildTemplate, evaluateWord } from './whole-token-template.js';


// Whole-token decode program (runtime.inference.session.decodeLoop.wholeTokenProgram).
//
// The ordinary recorded decode step is encoded four consecutive times (from decode
// step 2 of a generation) while every GPU command it records is observed:
// dispatches, bind group descriptors, uniform contents, queue writes and copies.
// The observations must agree on structure (pipelines, workgroup counts, bound
// buffers).
//
// Per-step inputs are declared by the operations that own them, on the recorder
// (see declareStepContract): the embedding gather names the queue write that
// carries the input token id, and operations whose commands change at a context
// threshold (KV capacity, attention variant caps, sliding-window fill) name the
// step positions over which their commands stay fixed or affine. Every other
// u32 word that differs between observations must be affine in the position, and
// every later observation must match the prediction made from the first two.
// Observations whose declared ranges or bound resources disagree start the
// observation over; any other difference rejects the program with an error
// (fail closed).
//
// The accepted program owns its bind groups, one packed uniform buffer and
// private copies of the step's transient buffers, all built once. It runs only
// at positions inside the declared ranges and the bounds of its affine words and
// copy offsets, with the plan and resources it was observed with; otherwise the
// caller invalidates it and observes again. A replayed token writes its
// position-dependent uniform words and token id with queue writes, submits one
// command buffer (one compute pass between copies) and maps back only the
// sampled token id. The next token's command buffer is encoded while the GPU
// runs the current one. The math is the recorded step's: the same pipelines,
// buffers and dispatch order.


/** @typedef {import('../../../../gpu/command-recorder.js').CommandRecorder} CommandRecorder */
/** @typedef {import('./whole-token-observation.js').WordModel} WordModel */
/** @typedef {import('./whole-token-observation.js').DynamicWord} DynamicWord */
/** @typedef {import('./whole-token-observation.js').BindGroupDescriptors} BindGroupDescriptors */
/** @typedef {import('./whole-token-observation.js').PositionLimit} PositionLimit */
/** @typedef {import('./whole-token-observation.js').SavedPatch} SavedPatch */
/** @typedef {import('./whole-token-template.js').BufferKey} BufferKey */
/** @typedef {import('./whole-token-template.js').UniformSlot} UniformSlot */
/** @typedef {import('./whole-token-template.js').TemplateStep} TemplateStep */
/** @typedef {import('./whole-token-template.js').ReadbackTemplate} ReadbackTemplate */
/** @typedef {import('./whole-token-template.js').TransientInfo} TransientInfo */

/** @typedef {{ kind: 'copy', src: GPUBuffer, srcOffset: WordModel, dst: GPUBuffer, dstOffset: WordModel, size: number }} ReplayCopy */
/** @typedef {{ kind: 'clear', buffer: GPUBuffer, offset: number, size: number }} ReplayClear */
/** @typedef {{ kind: 'dispatch', pipeline: GPUComputePipeline, groups: (GPUBindGroup | null)[], workgroups: number[] }} ReplayDispatch */
/** @typedef {{ kind: 'indirect', pipeline: GPUComputePipeline, groups: (GPUBindGroup | null)[], indirect: GPUBuffer, indirectOffset: number }} ReplayIndirect */
/** @typedef {ReplayCopy | ReplayClear | ReplayDispatch | ReplayIndirect} ReplayStep */
/** @typedef {{ buffer: GPUBuffer, offset: number, bytes: Uint8Array, dynamic: DynamicWord[] }} ReplayWrite */
/** @typedef {{ src: GPUBuffer, srcOffset: number, dstOffset: number, size: number }} ReplayReadbackCopy */
/** @typedef {{ index: number, model: WordModel }} DynamicUniformWord */
/** @typedef {{ pos: number, staging: GPUBuffer, commandBuffer: GPUCommandBuffer }} EncodedToken */

export const WHOLE_TOKEN_OBSERVED_STEPS = 4;

const U32_MAX = 0xffffffff;
// Written to the sampled-token word of a staging buffer before each submit; still
// there after the map means the command buffer never ran its readback copy.
const NO_TOKEN = new Uint32Array([U32_MAX]);

/**
 * @param {number} value
 * @param {number} alignment
 */
function alignUp(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

/**
 * @param {readonly unknown[]} a
 * @param {readonly unknown[]} b
 */
function sameIdentity(a, b) {
  return a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
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
    this.#watchBindGroups();
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
    this.stats = WholeTokenProgram.#freshStats();
    // The observations a built program came from; a later generation's first observed
    // step must agree with them (verify) before the kept program replays it.
    /** @type {StepObservation[]} */
    this.basis = [];
    this.awaitingVerification = false;
    // Built by #build and #materialize once the observed steps are accepted (declarations only).
    /** Step positions the program may run, and the operation that bounds them. @type {PositionLimit} */ this.positionRange;
    /** Resources the program was observed with. @type {readonly unknown[]} */ this.identity;
    /** Pooled buffers the program binds without owning them. @type {GPUBuffer[]} */ this.pooledExternals;
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

  static #freshStats() {
    return { observedSteps: 0, replayedTokens: 0, restarts: 0, invalidations: 0, reused: false, verifiedAhead: false,
      lastInvalidation: /** @type {string | null} */ (null) };
  }

  #watchBindGroups() {
    const device = this.device;
    patchMethod(this.watch, device, 'createBindGroup', (previous) => (/** @type {GPUBindGroupDescriptor} */ desc) => {
      const bindGroup = previous.call(device, desc);
      this.descriptors.set(bindGroup, {
        layout: desc.layout,
        label: desc.label ?? null,
        entries: Array.from(desc.entries, (entry) => ({ binding: entry.binding, resource: entry.resource })),
      });
      return bindGroup;
    });
  }

  /**
   * Observe one ordinary recorded decode step. `run` executes it and returns the token;
   * `identityOf` names the plan and resources the step ran with, read after it ran;
   * `limits` are position ranges the caller declares for the step, as its operations do.
   * @template T
   * @param {number} pos
   * @param {number} token
   * @param {(attach: (recorder: CommandRecorder | null) => void) => Promise<T>} run
   * @param {() => readonly unknown[]} identityOf
   * @param {readonly PositionLimit[]} [limits]
   * @returns {Promise<T>}
   */
  async observe(pos, token, run, identityOf, limits = []) {
    if (this.ready) throw new Error('[WholeTokenProgram] Program is already built.');
    const { observation, result } = await this.#capture(pos, token, run, identityOf, limits);
    this.#accept(observation);
    return result;
  }

  /**
   * Keep the built program for the next generation. Its first step (verify) must agree
   * with the steps the program was built from before it replays again.
   */
  beginGeneration() {
    if (!this.ready) throw new Error('[WholeTokenProgram] Only a built program can be kept.');
    this.pending = null;
    this.awaitingVerification = true;
    this.stats = WholeTokenProgram.#freshStats();
    this.#watchBindGroups();
  }

  /**
   * The kept program's first step of a generation: observe the ordinary recorded step and
   * accept the program only when that step builds the same template together with the
   * steps the program was built from (same commands, resources and bindings; constant
   * words equal, affine words on their lines at this position, the token declared), with
   * the same resources and inside both position ranges. Otherwise the program is
   * invalidated and this step starts a new observation window.
   * @template T
   * @param {number} pos
   * @param {number} token
   * @param {(attach: (recorder: CommandRecorder | null) => void) => Promise<T>} run
   * @param {() => readonly unknown[]} identityOf
   * @param {readonly PositionLimit[]} [limits]
   * @returns {Promise<T>}
   */
  async verify(pos, token, run, identityOf, limits = []) {
    if (!this.ready || !this.awaitingVerification) throw new Error('[WholeTokenProgram] No kept program awaits verification.');
    const { observation, result } = await this.#capture(pos, token, run, identityOf, limits);
    this.awaitingVerification = false;
    this.#unwatch();
    const reason = this.#adoptVerified(pos, observation, pos + 1);
    if (reason !== null) {
      this.invalidate(`kept program does not match position ${pos}: ${reason}`);
      this.#accept(observation);
    }
    return result;
  }

  /**
   * Verify the kept program on an ordinary step that is recorded but never submitted: the
   * same check as verify, for the step at `pos` that has not run yet. `run` records the
   * step and stops before its submit by throwing `stopSignal`; any other error propagates.
   * A match replays from `pos`; a mismatch invalidates the program and discards the
   * observation (a step that never ran observes nothing of the sequence).
   * @param {number} pos
   * @param {number} token
   * @param {(attach: (recorder: CommandRecorder | null) => void) => Promise<unknown>} run
   * @param {() => readonly unknown[]} identityOf
   * @param {readonly PositionLimit[]} limits
   * @param {unknown} stopSignal
   * @returns {Promise<boolean>}
   */
  async verifyAhead(pos, token, run, identityOf, limits, stopSignal) {
    if (!this.ready || !this.awaitingVerification) throw new Error('[WholeTokenProgram] No kept program awaits verification.');
    const { observation } = await this.#capture(pos, token, async (attach) => {
      try {
        await run(attach);
      } catch (error) {
        if (error === stopSignal) return;
        throw error;
      }
      throw new Error('[WholeTokenProgram] A step verified ahead must stop before its submit.');
    }, identityOf, limits);
    this.awaitingVerification = false;
    this.#unwatch();
    const reason = this.#adoptVerified(pos, observation, pos);
    if (reason !== null) this.invalidate(`kept program does not match position ${pos} ahead of time: ${reason}`);
    this.stats.verifiedAhead = reason === null;
    return reason === null;
  }

  /**
   * Accept the kept program for a generation whose observed step at `pos` builds the same
   * template as its basis, with the same resources and inside both position ranges; the
   * program then replays from `nextPos`. Returns why it does not, or null.
   * @param {number} pos
   * @param {StepObservation} observation
   * @param {number} nextPos
   * @returns {string | null}
   */
  #adoptVerified(pos, observation, nextPos) {
    const range = /** @type {PositionLimit} */ (observation.rangeLimit);
    const min = Math.max(this.positionRange.min, range.min);
    const max = Math.min(this.positionRange.max, range.max);
    if (!sameIdentity(observation.identity, this.identity)) return 'the execution plan or a bound resource changed';
    if (pos < min || pos > max) return `position ${pos} is outside [${min}, ${max}]`;
    if (this.pooledExternals.some((buffer) => !isBufferActive(buffer))) return 'a bound buffer returned to the buffer pool';
    try { buildTemplate([...this.basis, observation]); } catch (error) { return /** @type {Error} */ (error).message; }
    this.positionRange = { owner: max < this.positionRange.max ? range.owner : this.positionRange.owner, min, max };
    this.nextPos = nextPos;
    this.stats.reused = true;
    return null;
  }

  /**
   * Run one ordinary recorded decode step under observation.
   * @template T
   * @param {number} pos
   * @param {number} token
   * @param {(attach: (recorder: CommandRecorder | null) => void) => Promise<T>} run
   * @param {() => readonly unknown[]} identityOf
   * @param {readonly PositionLimit[]} limits
   * @returns {Promise<{ observation: StepObservation, result: T }>}
   */
  async #capture(pos, token, run, identityOf, limits) {
    const observation = new StepObservation(pos, token);
    /** @type {(() => void) | null} */
    let stop = null;
    const attach = (/** @type {CommandRecorder | null} */ recorder) => {
      if (!recorder) throw new Error('[WholeTokenProgram] Requires a recorded decode step.');
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
    for (const input of observation.pendingTokenInputs) {
      observation.problem(`a token input was declared for "${input.buffer.label ?? ''}" but never written`);
    }
    observation.identity = identityOf();
    observation.limits.push(...limits);
    observation.rangeLimit = observation.range();
    if (observation.problems.length > 0) {
      this.#unwatch();
      throw new Error(`[WholeTokenProgram] Decode step at position ${pos} cannot become a token program: ${observation.problems.join('; ')}.`);
    }
    return { observation, result };
  }

  /** @param {StepObservation} observation */
  #accept(observation) {
    this.observations.push(observation);
    this.stats.observedSteps++;
    const consistent = this.#consistentSuffix();
    if (consistent < this.observations.length) {
      // A declared threshold or a bound resource changed inside the window: start over from it.
      this.stats.restarts++;
      this.observations = this.observations.slice(this.observations.length - consistent);
    }
    if (this.observations.length === WHOLE_TOKEN_OBSERVED_STEPS) {
      this.#unwatch();
      this.#build();
      this.basis = this.observations;
      this.observations = [];
      this.nextPos = observation.pos + 1;
    }
  }

  /**
   * Number of newest observations that share the newest one's resources and all lie
   * inside the intersection of their declared position ranges.
   */
  #consistentSuffix() {
    const obs = this.observations;
    const newest = obs[obs.length - 1];
    let min = -Infinity;
    let max = Infinity;
    let count = 0;
    for (let i = obs.length - 1; i >= 0; i--) {
      const range = /** @type {PositionLimit} */ (obs[i].rangeLimit);
      if (!sameIdentity(obs[i].identity, newest.identity)) break;
      const nextMin = Math.max(min, range.min);
      const nextMax = Math.min(max, range.max);
      if (obs.slice(i).some((o) => o.pos < nextMin || o.pos > nextMax)) break;
      min = nextMin;
      max = nextMax;
      count++;
    }
    return count;
  }

  /**
   * Why the built program cannot run the step at `pos` with the resources `identity`
   * names, or null when it can.
   * @param {number} pos
   * @param {readonly unknown[]} identity
   * @returns {string | null}
   */
  rejects(pos, identity) {
    if (!this.ready) return 'the program is not built';
    if (pos !== this.nextPos) return `position ${pos} is not the next position ${this.nextPos}`;
    const range = this.positionRange;
    if (pos < range.min || pos > range.max) {
      return `position ${pos} is outside [${range.min}, ${range.max}] (bounded by ${range.owner})`;
    }
    if (!sameIdentity(identity, this.identity)) return 'the execution plan or a bound resource changed';
    for (const buffer of this.pooledExternals) {
      if (!isBufferActive(buffer)) return `bound buffer "${buffer.label ?? ''}" returned to the buffer pool`;
    }
    return null;
  }

  /**
   * Drop the built program so the following steps are observed again.
   * @param {string} reason
   */
  invalidate(reason) {
    this.#release();
    this.observations = [];
    this.stats.invalidations++;
    this.stats.lastInvalidation = reason;
    log.info('Decode', `Whole-token program invalidated: ${reason}; observing again.`);
    this.#watchBindGroups();
  }

  #build() {
    const built = buildTemplate(this.observations);
    this.identity = built.identity;
    const minAlign = this.device.limits?.minUniformBufferOffsetAlignment ?? 256;
    this.#materialize(built.template, built.uniformSlots, built.transientTemplate, built.readback, minAlign, built.declared);
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
   * @param {PositionLimit} declared
   */
  #materialize(template, uniformSlots, transientTemplate, readback, minAlign, declared) {
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

    // Affine words wrap past u32 and affine copies leave their buffers at some position;
    // the program runs only before the first such position.
    /** @type {PositionLimit} */
    let range = declared;
    /**
     * @param {string} owner
     * @param {number} last
     */
    const bound = (owner, last) => {
      if (last < range.max) range = { owner, min: range.min, max: last };
    };
    for (const { model } of this.dynamicUniformWords) bound('uniform word bounds', affineLastPosition(model, 0, U32_MAX));
    for (const write of this.writes) {
      for (const { model } of write.dynamic) bound('queue write word bounds', affineLastPosition(model, 0, U32_MAX));
    }
    for (const step of this.steps) {
      if (step.kind !== 'copy') continue;
      bound(`copy from "${step.src.label ?? ''}"`, affineLastPosition(step.srcOffset, 0, step.src.size - step.size));
      bound(`copy to "${step.dst.label ?? ''}"`, affineLastPosition(step.dstOffset, 0, step.dst.size - step.size));
    }
    this.positionRange = range;

    /** @type {Set<GPUBuffer>} */
    const external = new Set();
    for (const step of this.steps) {
      if (step.kind === 'copy') { external.add(step.src); external.add(step.dst); }
      else if (step.kind === 'clear') external.add(step.buffer);
      else if (step.kind === 'indirect') external.add(step.indirect);
    }
    for (const step of template) {
      if (step.kind !== 'dispatch' && step.kind !== 'indirect') continue;
      for (const group of step.groups) {
        for (const entry of group?.entries ?? []) if (entry.key !== undefined && typeof entry.key !== 'string') external.add(entry.key);
      }
    }
    for (const write of this.writes) external.add(write.buffer);
    for (const copy of this.readback) external.add(copy.src);
    for (const buffer of this.owned) external.delete(buffer);
    this.pooledExternals = [...external].filter((buffer) => isBufferActive(buffer));

    this.ready = true;
    log.info('Decode', `Whole-token program: ${this.dispatchCount} dispatches, ${uniformSlots.length} uniform slots `
      + `(${dynamicSlots.length} position/token dependent), ${substitutes.size} private transients, ${this.writes.length} queue writes, `
      + `${this.positionDependentCopies ?? 0} position-dependent copies; positions [${range.min}, ${range.max}] (bounded by ${range.owner}).`);
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
    this.device.queue.writeBuffer(job.staging, 0, NO_TOKEN);
    this.device.queue.submit([job.commandBuffer]);
    const mapped = job.staging.mapAsync(GPUMapMode.READ, 0, this.staging[0].size);
    this.stagingIndex = this.staging.indexOf(job.staging) ^ 1;
    this.pending = this.#encode(pos + 1, this.stagingIndex);
    await mapped;
    const words = new Uint32Array(job.staging.getMappedRange(0, this.staging[0].size)).slice();
    job.staging.unmap();
    if (words[0] === U32_MAX) {
      throw new Error(`[WholeTokenProgram] The token program at position ${pos} wrote no sampled token; its command buffer did not run.`);
    }
    this.nextPos = pos + 1;
    this.stats.replayedTokens++;
    return words;
  }

  #unwatch() {
    restorePatches(this.watch);
    this.watch = [];
  }

  #release() {
    this.pending = null;
    this.basis = [];
    this.awaitingVerification = false;
    for (const buffer of this.owned) buffer.destroy();
    this.owned = [];
    this.ready = false;
    this.positionDependentCopies = undefined;
  }

  destroy() {
    this.#unwatch();
    this.#release();
  }
}
