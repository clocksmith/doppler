import { isMapRead, U32 } from './whole-token-observation.js';

// Word classification and the template a whole-token program is built from
// (whole-token-program.js). The observed decode steps must agree command by command
// on kind, pipeline, geometry and bound buffers (per-step transients compared by
// first-use order); the declared token input becomes a token word, every other
// differing u32 word must be affine in the position, and the declared position
// ranges of all steps are intersected. Anything else throws (fail closed).

/** @typedef {import('./whole-token-observation.js').WordModel} WordModel */
/** @typedef {import('./whole-token-observation.js').DynamicWord} DynamicWord */
/** @typedef {import('./whole-token-observation.js').ObservedOp} ObservedOp */
/** @typedef {import('./whole-token-observation.js').PositionLimit} PositionLimit */
/** @typedef {import('./whole-token-observation.js').StepObservation} StepObservation */

/**
 * A buffer as recorded in the template: the buffer itself, or `transient:<n>` for a
 * per-step transient that the program replaces with a private copy.
 * @typedef {GPUBuffer | string} BufferKey
 */
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
 *   token: boolean,
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
/**
 * Classify one undeclared u32 word across the observations: constant, or affine in
 * the position. The slope comes from the first two observations, which must be
 * consecutive positions; every observation, consecutive or not (a kept program's
 * basis plus a later generation's step), must then lie on that line modulo 2^32.
 * The input token is never inferred from values; its owner declares it.
 * @param {readonly number[]} values
 * @param {readonly number[]} positions
 * @param {boolean} [variableAdvance] Explicit GPU windows may advance by different positive integer counts.
 * @returns {{ kind: 'const', value: number } | { kind: 'affine', base: number, slope: number, pos0: number } | null}
 */
export function classifyWord(values, positions, variableAdvance = false) {
  if (values.every((v) => v === values[0])) return { kind: 'const', value: values[0] };
  if (variableAdvance) {
    if (values.length !== positions.length || positions.length < 2
      || !positions.every(Number.isSafeInteger) || positions[1] <= positions[0]) return null;
    const slope = (values[1] - values[0]) / (positions[1] - positions[0]);
    if (!Number.isSafeInteger(slope) || !values.every((value, i) =>
      Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff
      && value === values[0] + slope * (positions[i] - positions[0]))) return null;
    return { kind: 'affine', base: values[0], slope, pos0: positions[0] };
  }
  if (positions.length < 2 || positions[1] - positions[0] !== 1) return null;
  const slope = (values[1] - values[0]) | 0;
  for (let i = 2; i < values.length; i++) {
    if (((values[i] - values[0]) | 0) !== (Math.imul(slope, positions[i] - positions[0]) | 0)) return null;
  }
  return { kind: 'affine', base: values[0], slope, pos0: positions[0] };
}

/**
 * @param {WordModel} model
 * @param {number} pos
 * @param {number} token
 * @returns {number}
 */
export function evaluateWord(model, pos, token) {
  if (model.kind === 'const') return model.value;
  if (model.kind === 'token') return token >>> 0;
  return (model.base + model.slope * (pos - model.pos0)) >>> 0;
}

/**
 * Last position (from the first observed one on) at which an affine word stays in
 * [lo, hi]; Infinity for a word that does not change.
 * @param {WordModel} model
 * @param {number} lo
 * @param {number} hi
 */
export function affineLastPosition(model, lo, hi) {
  if (model.kind !== 'affine' || model.slope === 0) return Infinity;
  const room = model.slope > 0 ? (hi - model.base) / model.slope : (model.base - lo) / -model.slope;
  return model.pos0 + Math.floor(room);
}

/**
 * Compare the observed byte arrays word by word. Returns dynamic word models or throws.
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
    if (values.every((v, n) => v === (tokens[n] >>> 0))) {
      throw new Error(
        `[WholeTokenProgram] ${where}: word ${i} (${values.join(', ')}) tracks the input token but no operation declared it a token input.`
      );
    }
    const model = classifyWord(values, positions, observations[0].gpuTokenInput !== null);
    if (!model) {
      throw new Error(
        `[WholeTokenProgram] ${where}: word ${i} (${values.join(', ')}) is not affine in the position and no operation declared it.`
      );
    }
    dynamic.push({ word: i, model });
  }
  return dynamic;
}

/**
 * @param {StepObservation} observation
 * @returns {(buffer: GPUBuffer) => BufferKey}
 */
function canonicalBuffers(observation) {
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

/**
 * The command template the observed steps agree on, with the uniform slots it reads,
 * the per-step transients it replaces, its readback copies, the intersection of the
 * steps' declared position ranges and the resources the last step ran with.
 * @param {StepObservation[]} obs
 */
export function buildTemplate(obs) {
  const gpuTokenInput = obs[0].gpuTokenInput;
  if (obs.some(o => o.gpuTokenInput !== gpuTokenInput)) throw new Error('[WholeTokenProgram] Token input contract changed.');
  const lengths = obs.map((o) => o.ops.length);
  if (lengths.some((n) => n !== lengths[0])) {
    throw new Error(`[WholeTokenProgram] Decode steps recorded ${lengths.join('/')} commands; the structure changes with position.`);
  }
  const canon = obs.map((o) => canonicalBuffers(o));
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
      if (ops.some((op) => op.token !== ops[0].token)) {
        throw new Error(`[WholeTokenProgram] Command ${i} is a declared token input in some decode steps only.`);
      }
      /** @type {DynamicWord[]} */
      const dynamic = ops[0].token
        ? [{ word: 0, model: { kind: 'token' } }]
        : classifyBytes(ops.map((op) => op.bytes), obs, `command ${i} queue write`);
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
        const model = classifyWord(ops.map((op) => op[field]), obs.map((o) => o.pos), gpuTokenInput !== null);
        if (!model) {
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
  const cpuToken = template.some(step => step.kind === 'write' && step.dynamic.some(d => d.model.kind === 'token'));
  if (gpuTokenInput !== null) {
    if (cpuToken || obs.some(o => o.transient.has(gpuTokenInput))
      || template.some(step => step.kind === 'write' && step.key === gpuTokenInput)
      || !template.some(step => (step.kind === 'copy' && step.src === gpuTokenInput)
        || ((step.kind === 'dispatch' || step.kind === 'indirect')
          && step.groups.some(group => group?.entries?.some(entry => entry.key === gpuTokenInput))))) {
      throw new Error('[WholeTokenProgram] GPU windows require a borrowed resident token input read by their commands, without CPU token writes.');
    }
  } else if (!cpuToken) {
    throw new Error('[WholeTokenProgram] No operation declared the decode step\'s input token; the program cannot feed it.');
  }
  // A uniform rewritten each step must reach its dispatches through an observed bind
  // group; one bound only by a group cached before observation would replay stale.
  for (const o of obs) {
    for (const buffer of o.uniformBytes.keys()) {
      if (!o.boundUniforms.has(buffer)) {
        throw new Error(`[WholeTokenProgram] Uniform buffer "${buffer.label ?? ''}" is written during the decode step `
          + 'but no observed bind group reads it.');
      }
    }
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

  /** @type {PositionLimit} */
  let declared = { owner: 'none', min: -Infinity, max: Infinity };
  for (const o of obs) {
    const range = /** @type {PositionLimit} */ (o.rangeLimit);
    declared = {
      owner: range.max < declared.max ? range.owner : declared.owner,
      min: Math.max(declared.min, range.min),
      max: Math.min(declared.max, range.max),
    };
  }
  return { template, uniformSlots, transientTemplate, readback, declared, identity: obs[obs.length - 1].identity };
}
