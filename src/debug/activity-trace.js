import { decodeSnapshotData, snapshotElementBytes } from './tensor.js';
import stages from '../config/inspection/activity-stages.json' with { type: 'json' };

const ROLES = new Set(stages.roles);
// Fixed per-layer parameters: one capture per layer serves every token.
const CONSTANT_ROLES = new Set(stages.constantRoles);
// Later tokens read these from every earlier token, so a token range keeps its prefix.
const PREFIX_ROLES = new Set(stages.prefixRoles);

/** @param {string} label @param {unknown} value @param {(item: unknown) => boolean} valid @returns {asserts value is unknown[] | null} */
function checkList(label, value, valid) {
  if (value === null) return;
  if (!Array.isArray(value) || new Set(value).size !== value.length || !value.every(valid)) {
    throw new Error(`Activity capture ${label} must be null (no restriction) or a list of distinct valid entries; an empty list records none.`);
  }
}

/** @param {unknown} selection @returns {asserts selection is import('./activity-trace.js').ActivitySelection} */
function checkSelection(selection) {
  const value = /** @type {Record<string, unknown> | null} */ (selection);
  if (!value || typeof value !== 'object' || ['roles', 'layers', 'tokens'].some(key => value[key] === undefined)) {
    throw new Error('Activity capture requires activitySelection with explicit roles, layers and tokens (null for no restriction).');
  }
  checkList('roles', value.roles, role => ROLES.has(/** @type {string} */ (role)));
  checkList('layers', value.layers, layer => Number.isSafeInteger(layer) && Number(layer) >= 0);
  const tokens = /** @type {{ from?: unknown; to?: unknown } | null} */ (value.tokens);
  if (tokens !== null && (!Number.isSafeInteger(tokens?.from) || Number(tokens.from) < 0
    || (tokens.to !== null && (!Number.isSafeInteger(tokens.to) || Number(tokens.to) <= Number(tokens.from))))) {
    throw new Error('Activity capture tokens must be null or { from, to } with to after from (to: null for no end).');
  }
}

/** @param {unknown} budgets @returns {asserts budgets is Record<string, number>} */
function checkRoleBudgets(budgets) {
  if (!budgets || typeof budgets !== 'object' || Array.isArray(budgets)) throw new Error('Activity capture requires activityRoleMaxBytes (an empty object for no per-kind budgets).');
  for (const [role, bytes] of Object.entries(budgets)) {
    if (!ROLES.has(role) || !Number.isSafeInteger(bytes) || bytes < 1) throw new Error(`Activity capture budget for "${role}" must name a capture kind and a positive byte count.`);
  }
}

// A request records less than its policy allows, or within smaller budgets;
// it never widens the policy. Anything outside the policy fails here, before a run.
/** @type {typeof import('./activity-trace.js').resolveActivityCapture} */
export function resolveActivityCapture(limits, request) {
  if (request === undefined || request === null) return limits;
  if (!limits?.activityMaxBytes) throw new Error('A capture selection requires an observation policy that records model activity.');
  if (typeof request !== 'object' || Array.isArray(request)) throw new Error('A capture selection must be an object.');
  const unknown = Object.keys(request).find(key => !['roles', 'layers', 'tokens', 'maxBytes', 'roleMaxBytes'].includes(key));
  if (unknown) throw new Error(`Unknown capture selection field "${unknown}".`);
  const policy = limits.activitySelection;
  checkSelection(policy);
  const asked = { roles: request.roles ?? null, layers: request.layers ?? null, tokens: request.tokens ?? null };
  checkSelection(asked);
  for (const key of /** @type {const} */ (['roles', 'layers'])) {
    const allowed = /** @type {unknown[] | null} */ (policy[key]);
    const outside = allowed ? (asked[key] ?? []).filter(item => !allowed.includes(item)) : [];
    if (outside.length) throw new Error(`This observation policy does not record ${key === 'roles' ? 'the kind' : 'layer'} "${outside[0]}".`);
  }
  if (policy.tokens && asked.tokens && (asked.tokens.from < policy.tokens.from
    || (policy.tokens.to !== null && (asked.tokens.to === null || asked.tokens.to > policy.tokens.to)))) {
    throw new Error('The requested token range reaches outside the range this observation policy records.');
  }
  const maxBytes = request.maxBytes ?? limits.activityMaxBytes;
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > limits.activityMaxBytes) {
    throw new Error(`A capture budget must be between 1 and the policy's ${limits.activityMaxBytes} bytes.`);
  }
  const roleMaxBytes = request.roleMaxBytes ?? {};
  checkRoleBudgets(roleMaxBytes);
  checkRoleBudgets(limits.activityRoleMaxBytes);
  for (const [role, bytes] of Object.entries(roleMaxBytes)) {
    const ceiling = Math.min(limits.activityRoleMaxBytes[role] ?? maxBytes, maxBytes);
    if (bytes > ceiling) throw new Error(`The budget for "${role}" exceeds the ${ceiling} bytes this policy and request allow.`);
  }
  return { ...limits, activityMaxBytes: maxBytes,
    activityRoleMaxBytes: { ...limits.activityRoleMaxBytes, ...roleMaxBytes },
    activitySelection: { roles: asked.roles ?? policy.roles, layers: asked.layers ?? policy.layers, tokens: asked.tokens ?? policy.tokens } };
}

/** @type {typeof import('./activity-trace.js').createActivityTrace} */
export function createActivityTrace(limits, started) {
  /** @type {Array<'activityMaxBytes' | 'activityMaxFrameValues' | 'activityMaxStateValues' | 'activityTokenWindow' | 'activitySummaryBlock'>} */
  const required = ['activityMaxBytes', 'activityMaxFrameValues', 'activityMaxStateValues', 'activityTokenWindow', 'activitySummaryBlock'];
  for (const name of required) {
    if (!Number.isSafeInteger(limits[name]) || limits[name] < 1) throw new Error(`Activity capture requires ${name}.`);
  }
  checkSelection(limits.activitySelection);
  checkRoleBudgets(limits.activityRoleMaxBytes);
  const selection = limits.activitySelection;
  const selectedRoles = selection.roles && new Set(selection.roles), selectedLayers = selection.layers && new Set(selection.layers);
  /** @type {Record<string, import('./activity-trace.js').ActivityCoverage>} */
  const coverage = {};
  const cover = (/** @type {string} */ role) => coverage[role] ??= { captured: 0, capturedBytes: 0, reused: 0, dropped: 0, failed: 0,
    notSelected: { role: 0, layer: 0, token: 0 }, firstDroppedToken: null, maxBytes: limits.activityRoleMaxBytes[role] ?? null };
  /** @type {Set<string>} */
  const constants = new Set();
  /** @type {import('./activity-trace.js').ActivityFrame[]} */
  const frames = [];
  /** @type {import('./activity-trace.js').ActivityWeight[]} */
  let weights = [];
  /** @type {Record<string, string>} */
  const unavailable = {};
  /** @type {Record<string, unknown>} */
  let model = {};
  /** @type {((buffer: GPUBuffer, offset: number, size: number) => Promise<ArrayBuffer>) | null} */
  let reader = null;
  let bytes = 0, forward = -1, tokenStart = 0;
  /** @type {'prefill' | 'decode'} */
  let phase = 'prefill';
  /** @type {number[]} */
  let tokenIds = [];
  /** @type {import('./activity-trace.js').ActivityTrace} */
  const api = {
    begin(nextPhase, start, ids) {
      if (nextPhase === phase && start === tokenStart && tokenIds.length && ids.join(',') === tokenIds.join(',')) return;
      phase = nextPhase; tokenStart = start; tokenIds = [...ids]; forward++;
    },
    setReader(value) { reader = value; },
    configure(config) { model = structuredClone(config); },
    async capture(role, buffer, options) {
      if (role === 'ffn-down-input' && !weights.some(w => w.layer === options.layer && w.columnEnergy.length)) return;
      if (!tokenIds.length) return;
      const seen = cover(role), constant = CONSTANT_ROLES.has(role), constantKey = `${role}/${options.layer}`;
      if (constants.has(constantKey)) { seen.reused++; return; }
      // What the selection leaves out is counted, not stored: it takes no frame and no budget.
      if (selectedRoles && !selectedRoles.has(role)) { seen.notSelected.role++; return; }
      if (selectedLayers && options.layer >= 0 && !selectedLayers.has(options.layer)) { seen.notSelected.layer++; return; }
      const originalShape = [...options.shape];
      let ids = [...(options.tokenIds ?? tokenIds)], start = options.tokenStart ?? tokenStart;
      let offset = options.offset ?? 0;
      const fullShape = [...originalShape];
      if (selection.tokens && !constant) {
        const first = Math.max(start, PREFIX_ROLES.has(role) ? 0 : selection.tokens.from);
        const end = Math.min(start + ids.length, selection.tokens.to ?? Infinity);
        if (first >= end) { seen.notSelected.token++; return; }
        // One row per token: keep only the selected rows. Other shapes are kept whole.
        if (!options.state && fullShape.length === 2 && fullShape[0] === ids.length && end - first < ids.length) {
          offset += (first - start) * fullShape[1]; ids = ids.slice(first - start, end - start); start = first; fullShape[0] = ids.length;
        }
      }
      if (frames.length >= limits.maxFrames) { unavailable['capture-limit'] = 'Activity frame limit reached; later boundaries are unrecorded.'; return; }
      if (role.startsWith('ffn-') && fullShape.length === 2 && fullShape[0] === ids.length && ids.length > limits.activityTokenWindow) {
        const skipped = ids.length - limits.activityTokenWindow;
        offset += skipped * fullShape[1]; start += skipped; ids = ids.slice(skipped); fullShape[0] = ids.length;
      }
      if (!fullShape.length || fullShape.some(n => !Number.isSafeInteger(n) || n <= 0)) throw new Error('Activity capture requires explicit positive dimensions.');
      const total = fullShape.reduce((a, b) => a * b, 1);
      const cap = options.state ? limits.activityMaxStateValues : limits.activityMaxFrameValues;
      const count = Math.min(total, cap);
      const elementBytes = snapshotElementBytes(options.dtype, options);
      const readCount = options.state ? Math.min(total, Math.floor(limits.maxFrameBytes / elementBytes)) : count;
      const summaryCount = options.state ? Math.ceil(readCount / limits.activitySummaryBlock) : 0;
      const payloadBytes = (count + summaryCount) * 8;
      /** @type {import('./activity-trace.js').ActivityFrame} */
      const frame = { index: frames.length, atMs: performance.now() - started, forward, phase, role,
        layer: options.layer, tokenStart: start,
        tokenIds: ids, shape: [count], fullShape, offset: 0,
        dtype: options.dtype, values: null, status: 'captured', reason: null,
        representation: options.representation ?? 'raw', metadata: { originalShape, precision: options.precision ?? null, ...options.metadata, ...(constant ? { constant: true } : {}) } };
      frames.push(frame);
      const overRole = seen.maxBytes !== null && seen.capturedBytes + payloadBytes > seen.maxBytes;
      if (overRole || bytes + payloadBytes > limits.activityMaxBytes) {
        frame.status = 'dropped';
        frame.reason = overRole ? `Activity budget for ${role} reached; inference continues.` : 'Activity recording budget reached; inference continues.';
        frame.metadata.limit = overRole ? 'activityRoleMaxBytes' : 'activityMaxBytes';
        seen.dropped++; seen.firstDroppedToken ??= start;
        return;
      }
      try {
        let values;
        if (buffer instanceof Float32Array || buffer instanceof Float64Array) {
          values = buffer.slice(offset, offset + readCount);
        } else {
          if (buffer.size < (offset + readCount) * elementBytes) throw new Error('Activity buffer is smaller than its declared range.');
          // Keep the recorder's resources alive, but submit the producing work
          // before copying. No completion promise can hang after recorder abort.
          options.recorder?.submitPartial();
          const byteOffset = offset * elementBytes;
          const alignedOffset = Math.floor(byteOffset / 4) * 4;
          const prefix = byteOffset - alignedOffset;
          const size = Math.ceil((prefix + readCount * elementBytes) / 4) * 4;
          if (!reader) throw new Error('GPU activity capture has no runtime-owned reader.');
          const raw = await reader(buffer, alignedOffset, size);
          const payload = raw.slice(prefix, prefix + readCount * elementBytes);
          values = options.dtype === 'u32' ? new Uint32Array(payload) : decodeSnapshotData(payload, options.dtype, options);
        }
        if (values.length !== readCount) throw new Error('Incomplete activity capture.');
        frame.values = Array.from(values.slice(0, count), v => Number.isFinite(v) ? v : /** @type {'NaN' | 'Infinity' | '-Infinity'} */ (String(v)));
        if (options.state) {
          /** @type {import('./activity-trace.js').ActivityNumber[]} */
          const summary = [];
          for (let start = 0; start < values.length; start += limits.activitySummaryBlock) {
            let sum = 0;
            const end = Math.min(values.length, start + limits.activitySummaryBlock);
            for (let i = start; i < end; i++) sum += values[i] * values[i];
            const rms = Math.sqrt(sum / (end - start));
            summary.push(Number.isFinite(rms) ? rms : /** @type {'NaN' | 'Infinity' | '-Infinity'} */ (String(rms)));
          }
          frame.summary = { blockSize: limits.activitySummaryBlock, observedValues: values.length, values: summary };
        }
        bytes += payloadBytes; seen.captured++; seen.capturedBytes += payloadBytes;
        if (constant) constants.add(constantKey);
      } catch (error) {
        frame.status = 'failed'; frame.reason = String(error); seen.failed++;
      }
    },
    weightAccess(layer, stage) {
      const roles = /** @type {Record<string, string[]>} */ (stages.weightRoles)[stage];
      const ids = weights.filter(w => w.layer === layer && roles?.includes(w.role)).map(w => w.id);
      if (!ids.length) return;
      void api.capture('weight-access', new Float32Array([1]), { layer, dtype: 'f32', shape: [1, 1],
        metadata: { weightIds: ids, stage, meaning: 'projection dispatch encoded' } });
    },
    async probe(stage, buffer, options) {
      const role = /** @type {Record<string,string>} */ (stages.probes)[stage];
      if (!role) return;
      await api.capture(role, buffer, { ...options, layer: options.layerIdx ?? -1,
        dtype: options.dtype ?? 'f32', shape: [options.numTokens, options.hiddenSize] });
    },
    unavailable(role, reason) { unavailable[role] = reason; },
    setWeights(value) { weights = value; },
    snapshot(includeFrames = true) {
      return { schema: 'doppler.model-activity/v1', frames: includeFrames ? [...frames] : [],
        weights, recordedBytes: bytes, limits: { ...limits }, selection: structuredClone(selection), coverage: structuredClone(coverage),
        unavailable: { ...unavailable }, model };
    },
    async logits(values) {
      await api.capture('logits', values, { layer: -1, dtype: 'f32', shape: [1, values.length],
        tokenStart: tokenStart + tokenIds.length - 1, tokenIds: tokenIds.slice(-1) });
    },
  };
  return api;
}
