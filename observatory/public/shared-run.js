// One run: its answer, the settings and hardware that produced it, its timing
// and whatever internals were recorded, under one identity. Everything here is
// a pure function over that object. There is no page, no GPU and no Doppler
// import, so a saved run can be opened, branched from and compared anywhere.

import { differenceStrength } from './viz/scale.js';

export const RUN_SCHEMA = 'doppler.observatory.run/v1';

// Sampling settings a replay must repeat for its tokens to mean the same thing.
const REPLAY_SAMPLING = ['temperature', 'topK', 'topP', 'repetitionPenalty', 'repetitionPenaltyWindow', 'presencePenalty', 'useChatTemplate'];
const MAX_RUN_TOKENS = 65536;
const MAX_LAYER_VALUES = 64 * 1024 * 1024;

const clone = (value) => (value == null ? null : JSON.parse(JSON.stringify(value)));
const isCount = (value) => Number.isSafeInteger(value) && value >= 0;
const isTokenList = (value) => Array.isArray(value) && value.length <= MAX_RUN_TOKENS && value.every(isCount);
const finite = (value) => (Number.isFinite(value) ? value : null);

async function digest(value) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(value)));
  return [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function toBase64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function fromBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// Layer outputs from a Doppler activation trace, packed as 32-bit floats in
// position → layer → component order. A position and layer that was not saved
// is listed in `missing`; it is never stored as zeros.
function packLayerOutputs(trace) {
  if (!trace?.frames?.length) return null;
  const { layers, width } = trace.geometry;
  const rows = new Map();
  for (const frame of trace.frames) {
    if (frame.role !== 'layer-output' || frame.status !== 'captured' || !frame.values) continue;
    for (let i = 0; i < frame.tokenIds.length; i++) {
      const key = `${frame.tokenStart + i}/${frame.layer}`;
      rows.set(key, { values: frame.values, offset: i * width });
    }
  }
  const positions = [...new Set([...rows.keys()].map((key) => Number(key.split('/')[0])))].sort((a, b) => a - b);
  if (!positions.length) return null;
  const data = new Float32Array(positions.length * layers * width);
  const missing = [];
  positions.forEach((position, p) => {
    for (let layer = 0; layer < layers; layer++) {
      const row = rows.get(`${position}/${layer}`);
      const base = (p * layers + layer) * width;
      if (!row) { missing.push([position, layer]); data.fill(NaN, base, base + width); continue; }
      for (let i = 0; i < width; i++) data[base + i] = Number(row.values[row.offset + i]);
    }
  });
  return { layers, width, positions, missing, encoding: 'f32-le-base64', data: toBase64(new Uint8Array(data.buffer)) };
}

// The vector one layer produced at one token position, or null when it was not saved.
// A position counts through everything the model was given and then what it
// answered, as the recorded layer outputs do. These convert to and from the
// index of an answer token.
export function sequencePosition(run, index) {
  return (run.inputTokenIds?.length ?? 0) + index;
}

export function answerIndex(run, position) {
  if (!Number.isInteger(position)) return null;
  const index = position - (run.inputTokenIds?.length ?? 0);
  return index >= 0 && index < run.tokens.length ? index : null;
}

export function layerVector(run, position, layer) {
  const saved = run.layerOutputs;
  if (!saved) return null;
  const p = saved.positions.indexOf(position);
  if (p < 0 || layer < 0 || layer >= saved.layers || saved.missing.some(([mp, ml]) => mp === position && ml === layer)) return null;
  saved.decoded ??= new Float32Array(fromBase64(saved.data).buffer);
  const base = (p * saved.layers + layer) * saved.width;
  return saved.decoded.subarray(base, base + saved.width);
}

// Per-step GPU time as the runtime measured it. Kernel labels such as
// "matmul:qkv_proj:L3" are summed across layers into "matmul:qkv_proj".
function gpuTiming(policy, stats) {
  if (!policy?.gpuTimestampQueries) return null;
  const steps = (stats?.decodeProfileSteps ?? []).filter((entry) => entry?.timings).map((entry) => {
    const kernels = {};
    for (const [label, ms] of Object.entries(entry.timings)) {
      if (!Number.isFinite(ms)) continue;
      const name = label.replace(/:L\d+$/, '');
      kernels[name] = (kernels[name] ?? 0) + ms;
    }
    return { step: entry.step, totalMs: Object.values(kernels).reduce((sum, ms) => sum + ms, 0), kernels };
  });
  // null means the runtime did not measure it. It is not a time of zero.
  return { decodeMs: finite(stats?.gpuTimeDecodeMs), prefillMs: finite(stats?.gpuTimePrefillMs), steps };
}

// What a run recorded besides its answer, and what it left out. The activity
// values themselves stay in the demo's recording file; a run keeps the account.
function captureAccount(receipt) {
  const trace = receipt.activationTrace ?? null;
  const activity = trace?.activity ?? null;
  const count = (frames, status) => (frames ?? []).filter((frame) => frame.status === status).length;
  return {
    layerOutputs: trace ? { captured: count(trace.frames, 'captured'), dropped: count(trace.frames, 'dropped'), failed: count(trace.frames, 'failed'), stoppedReason: trace.stoppedReason ?? null } : null,
    activity: activity ? {
      recordedBytes: activity.recordedBytes ?? null,
      maxBytes: activity.limits?.activityMaxBytes ?? trace.capturePolicy?.activityMaxBytes ?? null,
      captured: count(activity.frames, 'captured'),
      dropped: count(activity.frames, 'dropped'),
      failed: count(activity.frames, 'failed'),
      // Present when the runtime can record a chosen part of a run.
      selection: clone(activity.selection ?? null),
      coverage: clone(activity.coverage ?? null),
      unavailable: Object.keys(activity.unavailable ?? {}),
    } : null,
  };
}

// The recorded layer outputs at the position where the model chose an answer
// token: one before the token itself, since a position's last layer predicts
// the next token. Null when any layer there was not recorded; a projection of
// a partial stack would look complete and not be.
export function choiceVectors(run, index) {
  const position = sequencePosition(run, index) - 1;
  const saved = run.layerOutputs;
  if (!saved || position < 0 || !saved.positions.includes(position)) return null;
  const vectors = [];
  for (let layer = 0; layer < saved.layers; layer++) {
    const vector = layerVector(run, position, layer);
    if (!vector) return null;
    vectors.push(vector);
  }
  return { position, vectors };
}

// How much each layer changed the selected position's representation: the size
// of the layer's output and its distance from the layer before, relative to
// its own size. Measured outputs, a derived number. It says where the
// representation moved most, not which layer produced the answer. A layer whose
// output, or whose predecessor's, was not recorded has no change: null, not 0.
export function layerChanges(run, position) {
  const saved = run.layerOutputs;
  if (!saved || !saved.positions.includes(position)) return null;
  const rows = [];
  let before = null;
  for (let layer = 0; layer < saved.layers; layer++) {
    const vector = layerVector(run, position, layer);
    if (!vector) { rows.push({ layer, size: null, change: null }); before = null; continue; }
    let size = 0; let moved = 0;
    for (let i = 0; i < vector.length; i++) { size += vector[i] * vector[i]; if (before) { const d = vector[i] - before[i]; moved += d * d; } }
    size = Math.sqrt(size);
    rows.push({ layer, size, change: before && size > 0 ? Math.sqrt(moved) / size : null });
    before = vector;
  }
  return rows;
}

// What a run recorded and left out, in sentences. Missing data is named, never smoothed over.
export function describeCaptures(run) {
  const account = run.captures;
  if (!account || (!account.layerOutputs && !account.activity)) return ['No internals were recorded: this run has its answer, settings and timing only.'];
  const lines = [];
  const saved = run.layerOutputs;
  if (saved) lines.push(`Layer outputs for ${saved.positions.length} token positions × ${saved.layers} layers${saved.missing.length ? `; ${saved.missing.length} position and layer pairs were not saved` : ''}.`);
  else if (account.layerOutputs) lines.push('Layer outputs were requested but none were saved.');
  if (account.layerOutputs?.dropped) lines.push(`${account.layerOutputs.dropped} layer captures were dropped${account.layerOutputs.stoppedReason ? `: ${account.layerOutputs.stoppedReason}` : ''}.`);
  const activity = account.activity;
  if (activity) {
    const mib = (bytes) => (Number.isFinite(bytes) ? `${(bytes / 2 ** 20).toFixed(1)} MiB` : 'an unreported amount');
    lines.push(`Activity: ${activity.captured} captures, ${mib(activity.recordedBytes)} of a ${mib(activity.maxBytes)} budget.`);
    const kept = activity.selection;
    if (kept?.roles) lines.push(kept.roles.length ? `Recorded by choice: only ${kept.roles.join(', ')}.` : 'Recorded by choice: no activity kinds, layer outputs only.');
    if (kept?.layers) lines.push(`Recorded by choice: only layer${kept.layers.length === 1 ? '' : 's'} ${kept.layers.map((layer) => layer + 1).join(', ')}.`);
    if (kept?.tokens) lines.push(`Recorded by choice: only tokens ${kept.tokens.from}${kept.tokens.to === null ? ' onward' : `–${kept.tokens.to - 1}`}.`);
    const out = Object.entries(activity.coverage ?? {}).filter(([, c]) => c.dropped > 0);
    if (out.length) {
      const from = out.map(([, c]) => c.firstDroppedToken);
      const span = Math.min(...from) === Math.max(...from) ? `from token ${from[0]}` : `starting between tokens ${Math.min(...from)} and ${Math.max(...from)}`;
      lines.push(`Ran out of budget ${span}: ${out.map(([role]) => role).join(', ')}.`);
    }
    else if (activity.dropped) lines.push(`${activity.dropped} activity captures did not fit the budget and are missing.`);
    if (activity.failed) lines.push(`${activity.failed} activity captures failed.`);
  }
  return lines;
}

// The run a new run branched from, found by exact token IDs: a branch's input is
// its trunk's input, the trunk's answer up to the fork, and one different token.
export function findParent(run, runs) {
  const input = run.inputTokenIds;
  if (!input) return null;
  let best = null;
  for (const candidate of runs) {
    if (candidate.id === run.id || !candidate.inputTokenIds || candidate.model.manifestHash !== run.model.manifestHash) continue;
    const base = candidate.inputTokenIds;
    const index = input.length - base.length - 1;
    if (index < 0 || index >= candidate.tokens.length || !base.every((id, i) => input[i] === id)) continue;
    if (!candidate.tokens.slice(0, index).every((token, i) => input[base.length + i] === token.tokenId)) continue;
    const tokenId = input.at(-1);
    if (tokenId === candidate.tokens[index].tokenId) continue;
    // The longest matching trunk is the direct parent; shorter ones are ancestors.
    if (!best || base.length > best.inputLength) {
      const listed = candidate.tokens[index].candidates?.find((c) => c.tokenId === tokenId) ?? null;
      best = { inputLength: base.length, parent: { runId: candidate.id, index, replacedTokenId: candidate.tokens[index].tokenId, tokenId, tokenText: listed?.text ?? null, probability: listed?.probability ?? null } };
    }
  }
  return best?.parent ?? null;
}

// Builds a run from a Doppler inspection receipt.
export async function createRun({ receipt, prompt, environment = {}, parent = null, kind = null, createdAt = new Date().toISOString() }) {
  const identity = receipt.fingerprint?.identity ?? {};
  const evidence = receipt.generationEvidence ?? {};
  const stats = evidence.stats ?? {};
  const records = Array.isArray(receipt.tokens) && receipt.tokens.length === receipt.generatedTokenIds.length ? receipt.tokens : null;
  const run = {
    schema: RUN_SCHEMA,
    id: null,
    createdAt,
    // 'answer', 'branch', 'benchmark' or 'recorded-example'.
    kind: parent ? 'branch' : (kind ?? 'answer'),
    prompt: String(prompt ?? ''),
    parent: parent ? clone(parent) : null,
    model: { modelId: identity.artifact?.modelId ?? null, manifestHash: identity.artifact?.manifestHash ?? null, tokenizerDigest: identity.tokenizer?.digest ?? null },
    sampling: clone(evidence.generationConfig),
    observation: {
      policyId: receipt.policy?.id ?? null,
      modifiesExecution: receipt.policy?.modifiesExecution ?? null,
      performanceRepresentative: receipt.policy?.performanceRepresentative ?? null,
      captureSelection: clone(receipt.activationTrace?.activity?.selection ?? null),
    },
    environment: {
      runtimeVersion: environment.runtimeVersion ?? null,
      runtimeCommit: environment.runtimeCommit ?? null,
      // 'hosted', or 'preview' for an unqualified build; null in runs saved before this was recorded.
      runtimeChannel: environment.runtimeChannel ?? null,
      userAgent: environment.userAgent ?? null,
      adapter: clone(evidence.backendIdentity?.adapter ?? null),
      activationDtype: evidence.backendIdentity?.activationDtype ?? null,
      kernelPathId: evidence.backendIdentity?.kernelPathId ?? null,
    },
    // Exact IDs the model was given before its first token; null when the runtime did not report them.
    inputTokenIds: receipt.inputTokenIds ? [...receipt.inputTokenIds] : null,
    outputText: receipt.outputText ?? '',
    tokens: receipt.generatedTokenIds.map((tokenId, index) => ({
      index,
      tokenId,
      text: records?.[index]?.text ?? null,
      probability: finite(records?.[index]?.probability),
      // Entropy of the full next-token distribution, from runtimes that record it.
      entropy: finite(records?.[index]?.entropy),
      candidates: records ? (records[index].topCandidates ?? []).map((c) => ({ tokenId: c.tokenId, text: c.text, probability: c.probability })) : null,
    })),
    timing: {
      // Wall time from a tier that changes execution is not a speed measurement.
      representative: receipt.performanceRepresentative === true,
      wallMs: finite(receipt.wallTimingMs),
      timeToFirstTokenMs: finite(stats.ttftMs),
      prefillMs: finite(stats.prefillTimeMs),
      prefillTokens: stats.prefillTokens ?? null,
      decodeMs: finite(stats.decodeTimeMs),
      decodeTokens: stats.decodeTokens ?? null,
      stopReason: stats.stopReason ?? null,
      gpu: gpuTiming(receipt.policy, stats),
    },
    layerOutputs: packLayerOutputs(receipt.activationTrace),
    captures: captureAccount(receipt),
    // Keep detailed witnesses when requested; ordinary layer-only runs stay compact.
    activityTrace: receipt.activationTrace?.activity
      ? JSON.parse(JSON.stringify(receipt.activationTrace, (_, value) => ArrayBuffer.isView(value) ? Array.from(value) : value))
      : null,
  };
  run.id = `run-${(await digest({ ...run, layerOutputs: run.layerOutputs?.data ?? null })).slice(0, 16)}`;
  return run;
}

// Checks a run that came from a file. Throws with the reason a person can act on.
export function validateRun(run) {
  const fail = (why) => { throw new Error(`This is not a run this page can open: ${why}.`); };
  if (!run || typeof run !== 'object' || Array.isArray(run)) fail('the file does not hold a run');
  if (typeof run.schema !== 'string') fail('the file does not say what it holds');
  if (run.schema !== RUN_SCHEMA) fail(`its format is "${run.schema.slice(0, 60)}", not ${RUN_SCHEMA}`);
  if (typeof run.id !== 'string' || !/^run-[0-9a-f]{16}$/.test(run.id)) fail('it has no run identity');
  if (!run.model || !run.environment || !run.observation || !run.timing) fail('it is missing its model, environment, observation or timing');
  if (!(run.inputTokenIds === null || isTokenList(run.inputTokenIds))) fail('its input tokens are malformed');
  if (!Array.isArray(run.tokens) || run.tokens.length > MAX_RUN_TOKENS) fail('its answer tokens are malformed');
  run.tokens.forEach((token, index) => {
    if (token?.index !== index || !isCount(token.tokenId) || !(token.text === null || typeof token.text === 'string')
      || !(token.probability === null || (Number.isFinite(token.probability) && token.probability >= 0 && token.probability <= 1.000001))
      || !(token.entropy === undefined || token.entropy === null || (Number.isFinite(token.entropy) && token.entropy >= 0))
      || !(token.candidates === null || (Array.isArray(token.candidates) && token.candidates.length <= 256
        && token.candidates.every((c) => isCount(c?.tokenId) && typeof c.text === 'string' && Number.isFinite(c.probability) && c.probability >= 0 && c.probability <= 1.000001)
        && new Set(token.candidates.map((c) => c.tokenId)).size === token.candidates.length))) {
      fail(`answer token ${index} is malformed`);
    }
  });
  if (run.parent !== null && (typeof run.parent?.runId !== 'string' || !isCount(run.parent.index) || !isCount(run.parent.tokenId))) fail('its branch point is malformed');
  const saved = run.layerOutputs;
  if (saved !== null) {
    if (!saved || !isCount(saved.layers) || !isCount(saved.width) || !isTokenList(saved.positions) || !Array.isArray(saved.missing)
      || saved.encoding !== 'f32-le-base64' || typeof saved.data !== 'string') fail('its layer outputs are malformed');
    const values = saved.positions.length * saved.layers * saved.width;
    if (values > MAX_LAYER_VALUES || Math.ceil(values * 4 / 3) * 4 !== saved.data.length) fail('its layer outputs do not match their declared size');
  }
  // The demo's own activity recording, carried so a reopened run can restore
  // its grids. Its contents are checked by the demo when it is opened.
  const recording = run.activityRecording;
  if (recording !== undefined && recording !== null
    && (typeof recording !== 'object' || typeof recording.schema !== 'string' || !recording.schema.startsWith('doppler.activation-recording/') || !recording.trace)) {
    fail('its activity recording is malformed');
  }
  const trace = run.activityTrace;
  if (trace !== undefined && trace !== null && (typeof trace !== 'object' || !trace.geometry
    || !Array.isArray(trace.frames) || JSON.stringify(trace).length > 64 * 1024 * 1024)) {
    fail('its detailed activity trace is malformed or exceeds 64 MiB');
  }
  const gpu = run.timing.gpu;
  if (gpu !== null && (!Array.isArray(gpu?.steps) || gpu.steps.length > MAX_RUN_TOKENS || gpu.steps.some((s) => !Number.isFinite(s?.totalMs) || !s.kernels || typeof s.kernels !== 'object'))) {
    fail('its GPU timing is malformed');
  }
  return run;
}

// A run as a file: the decoded copy of the layer outputs is working state, not content.
export function serializeRun(run) {
  // `decoded`, `opened` and `recordingReleased` are this page's working state, not the run.
  return JSON.stringify(run, (key, value) => (['decoded', 'opened', 'recordingReleased'].includes(key) ? undefined : value));
}

// What the model could have said at one answer token: the listed candidates,
// which of them have been run as branches, and how much probability the list
// leaves out. Returns null when the run recorded no probabilities.
export function alternatives(run, index, runs = []) {
  const token = run.tokens[index];
  if (!token?.candidates) return null;
  const branches = runs.filter((other) => other.parent?.runId === run.id && other.parent.index === index);
  const listed = token.candidates.reduce((sum, c) => sum + c.probability, 0);
  return {
    candidates: token.candidates.map((c) => ({
      ...c,
      state: c.tokenId === token.tokenId ? 'taken' : branches.some((b) => b.parent.tokenId === c.tokenId) ? 'run' : 'not-run',
      branchRunId: branches.find((b) => b.parent.tokenId === c.tokenId)?.id ?? null,
    })),
    // Probability on tokens outside the list. They exist; they are just not shown.
    unlistedProbability: Math.max(0, 1 - listed),
    takenListed: token.candidates.some((c) => c.tokenId === token.tokenId),
  };
}

// The request that replays a run up to one answer token, puts a different token
// there, and continues. The prefix is the exact token IDs, never re-tokenized text.
export function forkRequest(run, index, tokenId) {
  const token = run.tokens[index];
  if (!token) throw new Error('There is no answer token at that position.');
  if (!run.inputTokenIds) throw new Error('This run did not keep the exact tokens it was given, so it cannot be continued from a chosen token.');
  if (tokenId === token.tokenId) throw new Error('That is the token the run already took.');
  const limit = run.sampling?.maxTokens;
  const remaining = Number.isSafeInteger(limit) ? limit - index - 1 : null;
  if (remaining === null) throw new Error('This run did not record its length limit, so a branch cannot be given the same one.');
  if (remaining < 1) throw new Error('The run\'s length limit leaves no room to continue after this token.');
  const generation = { maxTokens: remaining };
  for (const name of REPLAY_SAMPLING) if (run.sampling[name] !== undefined && run.sampling[name] !== null) generation[name] = run.sampling[name];
  const candidate = token.candidates?.find((c) => c.tokenId === tokenId) ?? null;
  return {
    inputTokenIds: [...run.inputTokenIds, ...run.tokens.slice(0, index).map((t) => t.tokenId), tokenId],
    generation,
    parent: { runId: run.id, index, replacedTokenId: token.tokenId, tokenId, tokenText: candidate?.text ?? null, probability: candidate?.probability ?? null },
  };
}

// A trunk and the branches run from it, nested. Branches of branches hang from their own parent.
export function forest(root, runs) {
  return {
    run: root,
    branches: runs.filter((other) => other.parent?.runId === root.id)
      .sort((a, b) => a.parent.index - b.parent.index || a.createdAt.localeCompare(b.createdAt))
      .map((child) => forest(child, runs)),
  };
}

const sameList = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((value, i) => value === b[i]);

function firstListDifference(a, b) {
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; i++) if (a[i] !== b[i]) return i;
  return null;
}

// Compares two saved runs of the same model on the same input. It reports the
// first place they were observed to differ, by answer token and by layer. That
// is where a difference first shows, not a finding about what caused it.
export function compareRuns(a, b, { tolerance = 1e-3 } = {}) {
  const blockers = [];
  if (!a.model.manifestHash || a.model.manifestHash !== b.model.manifestHash) blockers.push('The two runs used different model files.');
  if (a.model.tokenizerDigest !== b.model.tokenizerDigest) blockers.push('The two runs used different tokenizers.');
  if (!a.inputTokenIds || !b.inputTokenIds) blockers.push('At least one run did not keep the exact tokens it was given.');
  else if (!sameList(a.inputTokenIds, b.inputTokenIds)) blockers.push(`The two runs were given different input tokens (they first differ at input token ${firstListDifference(a.inputTokenIds, b.inputTokenIds) + 1}).`);
  const samplingFields = [...new Set([...Object.keys(a.sampling ?? {}), ...Object.keys(b.sampling ?? {})])]
    .filter((key) => JSON.stringify(a.sampling?.[key]) !== JSON.stringify(b.sampling?.[key]));
  if (samplingFields.length) blockers.push(`The two runs used different sampling settings (${samplingFields.join(', ')}).`);

  // What differs between the two runs' circumstances. These are what a comparison is for.
  const circumstances = [];
  const note = (field, left, right) => { if (JSON.stringify(left) !== JSON.stringify(right)) circumstances.push({ field, a: left ?? null, b: right ?? null }); };
  note('Runtime version', a.environment.runtimeVersion, b.environment.runtimeVersion);
  note('Runtime commit', a.environment.runtimeCommit, b.environment.runtimeCommit);
  note('Runtime channel', a.environment.runtimeChannel ?? null, b.environment.runtimeChannel ?? null);
  note('Graphics adapter', a.environment.adapter, b.environment.adapter);
  note('Browser', a.environment.userAgent, b.environment.userAgent);
  note('Activation precision', a.environment.activationDtype, b.environment.activationDtype);
  note('Kernel path', a.environment.kernelPathId, b.environment.kernelPathId);
  note('Observation tier', a.observation.policyId, b.observation.policyId);
  if (blockers.length) return { comparable: false, blockers, circumstances, tokens: null, layers: null };

  const idsA = a.tokens.map((t) => t.tokenId);
  const idsB = b.tokens.map((t) => t.tokenId);
  const firstToken = firstListDifference(idsA, idsB);
  const tokens = {
    lengths: [idsA.length, idsB.length],
    firstDifference: firstToken,
    a: firstToken === null ? null : a.tokens[firstToken] ?? null,
    b: firstToken === null ? null : b.tokens[firstToken] ?? null,
  };

  let layers = null;
  let layersUnavailable = null;
  const la = a.layerOutputs;
  const lb = b.layerOutputs;
  if (!la || !lb) layersUnavailable = 'At least one run did not record layer outputs, so the runs can only be compared by their tokens.';
  else if (la.layers !== lb.layers || la.width !== lb.width) layersUnavailable = 'The two runs recorded layer outputs of different shapes.';
  else {
    // Once the answers differ, later positions were given different tokens, so
    // their layer outputs are no longer the same question asked twice.
    const lastComparable = firstToken === null ? Infinity : a.inputTokenIds.length + firstToken - 1;
    const positions = la.positions.filter((position) => lb.positions.includes(position) && position <= lastComparable);
    const relative = [];
    let first = null;
    let firstAny = null;
    let unrecorded = 0;
    for (const position of positions) {
      const row = [];
      for (let layer = 0; layer < la.layers; layer++) {
        const va = layerVector(a, position, layer);
        const vb = layerVector(b, position, layer);
        if (!va || !vb) { row.push(null); unrecorded++; continue; }
        let diff = 0; let na = 0; let nb = 0;
        for (let i = 0; i < va.length; i++) { const d = va[i] - vb[i]; diff += d * d; na += va[i] * va[i]; nb += vb[i] * vb[i]; }
        const scale = Math.sqrt(Math.max(na, nb));
        const value = scale > 0 ? Math.sqrt(diff) / scale : (diff > 0 ? Infinity : 0);
        row.push(Number.isFinite(value) ? value : null);
        if (!Number.isFinite(value)) { unrecorded++; continue; }
        if (firstAny === null && value > 0) firstAny = { position, layer, relative: value };
        if (first === null && value > tolerance) first = { position, layer, relative: value };
      }
      relative.push(row);
    }
    layers = { positions, layerCount: la.layers, relative, tolerance, firstAboveTolerance: first, firstNonzero: firstAny, unrecorded };
  }
  return { comparable: true, blockers, circumstances, tokens, layers, layersUnavailable };
}

// A comparison ready to show: sentences and a grid, the same on every page.
export function describeComparison(a, b, { tolerance = 1e-3 } = {}) {
  const result = compareRuns(a, b, { tolerance });
  const text = (value) => (value === null || value === undefined ? 'not recorded' : typeof value === 'object' ? Object.values(value).filter(Boolean).join(' ') : String(value));
  const quote = (token) => (token?.text === null || token?.text === undefined ? '(text not recorded)' : token.text.replace(/\n/g, '⏎'));
  const share = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(value >= 0.1 ? 0 : 1)}%` : 'probability not recorded');
  const view = {
    comparable: result.comparable,
    blockers: result.blockers,
    circumstances: result.circumstances.map((row) => ({ field: row.field, a: text(row.a), b: text(row.b) })),
    verdict: 'These two runs are not the same question, so they cannot be compared.',
    // Random sampling without a fixed seed makes different answers expected, not a finding.
    sampling: Number(a.sampling?.temperature) > 0 && (a.sampling?.seed === null || a.sampling?.seed === undefined)
      ? `Both runs sampled at random (temperature ${a.sampling.temperature}, no fixed seed), so their answers can differ by chance. Where they do, the layer outputs before that token still show whether the model was in the same state.`
      : '',
    tokens: '',
    first: '',
    note: '',
    grid: null,
  };
  if (!result.comparable) return view;
  const { tokens, layers } = result;
  view.verdict = tokens.firstDifference === null && !layers?.firstAboveTolerance ? 'No difference observed.' : 'The runs differ.';
  view.tokens = tokens.firstDifference === null
    ? `Both runs produced the same ${tokens.lengths[0]} tokens.`
    : `The answers first differ at token ${tokens.firstDifference + 1}: "${quote(tokens.a)}" (${share(tokens.a?.probability)}) in the first run, "${quote(tokens.b)}" (${share(tokens.b?.probability)}) in the second.`;
  if (!layers) { view.note = result.layersUnavailable; return view; }
  const promptLength = a.inputTokenIds.length;
  const place = (position) => (position < promptLength ? `input token ${position + 1}` : `answer token ${position - promptLength + 1}`);
  const first = layers.firstAboveTolerance;
  view.first = first
    ? `First observed difference: layer ${first.layer + 1} at ${place(first.position)}, ${first.relative.toExponential(1)} relative.`
    : layers.firstNonzero
      ? `Nothing above ${layers.tolerance}. The first difference of any size is at layer ${layers.firstNonzero.layer + 1}, ${place(layers.firstNonzero.position)}: ${layers.firstNonzero.relative.toExponential(1)} relative.`
      : 'Every compared layer output is identical.';
  view.note = `${layers.positions.length} token positions × ${layers.layerCount} layers compared`
    + `${tokens.firstDifference === null ? '' : ', up to the last position where both runs were given the same tokens'}`
    + `${layers.unrecorded ? `; ${layers.unrecorded} cells were not recorded in one of the runs and are hatched` : ''}.`;
  view.grid = {
    columns: layers.positions.map((position) => ({ label: position < promptLength ? `i${position + 1}` : `a${position - promptLength + 1}`, title: place(position), position })),
    rows: Array.from({ length: layers.layerCount }, (_, layer) => ({
      label: `layer ${layer + 1}`,
      layer,
      cells: layers.positions.map((position, p) => {
        const value = layers.relative[p][layer];
        if (value === null) return { recorded: false, title: `${place(position)}, layer ${layer + 1}: not recorded` };
        return {
          recorded: true,
          value,
          first: Boolean(first && first.position === position && first.layer === layer),
          // Log scale from 1e-7 (faint) to 1 (solid): differences span many orders of magnitude.
          strength: differenceStrength(value),
          title: `${place(position)}, layer ${layer + 1}: ${value === 0 ? 'identical' : `${value.toExponential(1)} relative`}`,
        };
      }),
    })),
  };
  return view;
}

// Per-step GPU time for a run, or why there is none. Decode step s produces
// answer token s; the first answer token comes out of reading the prompt.
export function stepTimings(run) {
  const gpu = run.timing?.gpu;
  if (!gpu) return { available: false, reason: 'This run was recorded without GPU timing.' };
  if (!gpu.steps.length) return { available: false, reason: 'The runtime reported no timed steps for this run.' };
  const kernels = {};
  for (const step of gpu.steps) for (const [name, ms] of Object.entries(step.kernels)) kernels[name] = (kernels[name] ?? 0) + ms;
  return {
    available: true,
    steps: gpu.steps.map((step) => ({ ...step, tokenIndex: step.step, tokenText: run.tokens[step.step]?.text ?? null })),
    kernels: Object.entries(kernels).sort((x, y) => y[1] - x[1]).map(([name, ms]) => ({ name, ms })),
    decodeGpuMs: gpu.decodeMs,
    // Reading the prompt is not timed on the GPU by the runtime today.
    prefillGpuMs: gpu.prefillMs,
    decodeWallMs: run.timing.decodeMs,
  };
}

