// The workspace: one prompt, its answer streaming below, and every view of the
// run following one shared selection. Recording needs WebGPU; recorded runs and
// saved files open without it.

import { loadFlags } from './flags.js';
import { describeDevice } from './run.js';
import { mountNav } from './site.js';
import { take } from './handoff.js';
import { describeRuntime, loadRuntimeChoice } from './runtimes.js';
import { answerIndex, choiceVectors, createRun, forkRequest, sequencePosition, serializeRun, validateRun } from './shared-run.js';
import { createSelection } from './selection.js';
import { $, h, percent, visible } from './viz/dom.js';
import { answerSignals, DIVERSITY_WINDOW, perplexity } from './viz/signals.js';
import { renderForestTree, renderTokenPanel } from './views/forest.js';
import { renderJourney } from './views/journey.js';
import { renderActivity } from './views/activity.js';
import { renderAttention } from './views/attention.js';
import { renderMemory } from './views/memory.js';
import { renderGearbox } from './views/gearbox.js';
import { renderAtlas } from './views/atlas.js';
import { renderConcertina } from './views/concertina.js';
import { renderFaultline } from './views/faultline.js';
import { readRecordingBlob, readRecordingJson } from '/doppler/demo/ui/recording-import.js';
import { retain, redactedSummary, downloadFile, retainedBytes, retainedLimit } from './run-files.js';
import { assertRunModel, validateProjection } from './projection-contract.js';

const DEMO = '/doppler/demo';
const EXAMPLE_PROMPTS = ['Explain why the sky is blue in one sentence.', 'Write a haiku about rain.', 'What is 17 × 23?', 'Name three uses of a paperclip.'];
const VIEWS = [
  { id: 'activity', label: 'Activity', render: renderActivity },
  { id: 'journey', label: 'Layer journey', render: renderJourney },
  { id: 'attention', label: 'Attention', render: renderAttention },
  { id: 'memory', label: 'Memory (KV cache)', render: renderMemory },
  { id: 'gearbox', label: 'Gearbox (FFN)', render: renderGearbox },
  { id: 'forest', label: 'Branches', render: null },
  { id: 'atlas', label: 'Candidates', render: renderAtlas },
  { id: 'time', label: 'Time', render: renderConcertina },
];

const selection = createSelection();
const state = {
  runs: [],
  live: null, // { prompt, tokens } while an answer streams in
  view: 'activity',
  allViews: false,
  device: null,
  choice: null,
  runtime: null, // { dr, version, commit, channel, selection }
  model: null,
  busy: false,
  abort: null, // stops the answer being generated
  saved: new Set(),
  downloads: new Set(),
  retained: new Map(),
  partialSaved: false,
  partialDownload: false,
  importing: null,
  projections: new Map(),
  exampleRun: null,
};

const shown = () => state.runs.find((run) => run.id === selection.get().runId) ?? null;

// What a run's answer reads as. A branch inherits the words before its fork from
// the run it continues (with their recorded probabilities and candidates), puts
// one other word there, then generates its own. Display index k is sequence
// position base + k in the shown run.
function displayOf(run, seen = new Set()) {
  const own = run.tokens.map((token, i) => ({ ...token, owner: run, ownIndex: i }));
  const start = sequencePosition(run, 0);
  const parent = run.parent && !seen.has(run.id) ? state.runs.find((item) => item.id === run.parent.runId) : null;
  if (!parent || start === null) return { tokens: own, inherited: 0, base: start, parent: null };
  const above = displayOf(parent, new Set([...seen, run.id]));
  const forkAt = above.inherited + run.parent.index;
  const replaced = above.tokens[forkAt] ?? null;
  const fork = { index: forkAt, tokenId: run.parent.tokenId, text: run.parent.tokenText, probability: run.parent.probability ?? null,
    candidates: replaced?.candidates ?? null, owner: replaced?.owner ?? parent, ownIndex: replaced?.ownIndex ?? run.parent.index, inherited: true, forked: true };
  return { tokens: [...above.tokens.slice(0, forkAt).map((token) => ({ ...token, inherited: true })), fork, ...own], inherited: forkAt + 1, base: start - (forkAt + 1), parent };
}

// The display index of the selected position, or null.
function pickedIndex(run, display) {
  const position = selection.get().position;
  if (!run || position === null || display.base === null) return answerIndex(run, position);
  const k = position - display.base;
  return k >= 0 && k < display.tokens.length ? k : null;
}
// A stopped answer stays on screen only until another run is chosen.
const showRun = (runId) => { if (!leavePartial()) return; selection.select({ runId }); render(); };
const status = (text) => { $('ws-status').textContent = text; };

// ------------------------------------------------------------------ runtime

function leavePartial() {
  if (!state.live?.stopped) return true;
  if (state.live.tokens.length && !state.partialSaved && !confirm('Discard the unsaved partial answer?')) return false;
  state.live = null;
  return true;
}

async function loadRuntime() {
  const module = await import(state.choice.moduleUrl);
  const dr = module.dr ?? module.doppler;
  // Read the policy registry from this runtime before starting any model download.
  const policyModule = new URL('./client/inspection.js', new URL(state.choice.moduleUrl, location.href));
  const policies = (await import(policyModule.href)).listObservationPolicies();
  const details = (await dr.listModelDetails()).filter((entry) => entry?.modes?.includes('text') && entry?.classification?.tasks?.includes('generation'));
  let stored = new Set();
  try { stored = new Set((await dr.listPersistentModels()).map((entry) => entry.modelId)); } catch { /* shown as not saved */ }
  $('ws-model').replaceChildren(...details.map((entry) => h('option', { value: entry.modelId },
    `${entry.label ?? entry.modelId}${stored.has(entry.modelId) ? ' · saved' : entry.sizeBytes ? ` · ${(entry.sizeBytes / 1e6).toFixed(0)} MB` : ''}`)));
  const saved = details.find((entry) => stored.has(entry.modelId));
  if (saved) $('ws-model').value = saved.modelId;
  $('ws-model').disabled = false;
  state.runtime = { dr, version: module.DOPPLER_VERSION ?? null, commit: state.choice.release?.sourceCommit ?? null, channel: state.choice.id, details, stored, policies };
  syncMenuLabel();
}

function syncMenuLabel() {
  updateSend();
  const option = $('ws-model').selectedOptions[0];
  $('ws-menu-label').textContent = state.runtime && option ? `Next answer: ${option.textContent.split(' · ')[0]}` : 'Next answer model';
}

async function ensureModel(modelId, signal) {
  signal?.throwIfAborted();
  if (state.model?.modelId === modelId) return state.model.handle;
  if (state.model) { await state.model.handle.unload?.().catch(() => {}); state.model = null; }
  signal?.throwIfAborted();
  const handle = await state.runtime.dr.load(modelId, { cache: 'opfs', signal,
    onProgress: (event) => { if (!signal?.aborted) status(event?.message ?? 'Loading the model…'); } });
  state.runtime.stored.add(modelId);
  if (signal?.aborted) { await handle.unload(); signal.throwIfAborted(); }
  state.model = { modelId, handle };
  return handle;
}

// Which observation tier records what the menu asks for.
function observationFor({ layers, timing, witnesses }) {
  if (witnesses) {
    if (!state.runtime.policies.some((policy) => policy.id === 'demo/model-activity-timing' && policy.activationTrace?.activitySelection !== undefined)) throw new Error('this runtime cannot record attention, memory and weights');
    return { policyId: 'demo/model-activity-timing' };
  }
  if (layers && timing) {
    if (!state.runtime.policies.some((policy) => policy.id === 'demo/model-activity-timing' && policy.activationTrace?.activitySelection !== undefined)) throw new Error('this runtime cannot record layer outputs and GPU time in one run; choose one in the menu');
    return { policyId: 'demo/model-activity-timing', capture: { roles: [] } };
  }
  if (layers) return { policyId: 'demo/activation-grid' };
  if (timing) return { policyId: 'demo/deep-xray' };
  return { policyId: 'demo/guided-quality' };
}

const environment = () => ({ runtimeVersion: state.runtime.version, runtimeCommit: state.runtime.commit, runtimeChannel: state.runtime.channel, userAgent: navigator.userAgent });

// ------------------------------------------------------------------ asking

async function ask(prompt, captures = menuCaptures()) {
  if (!prompt || state.busy || state.importing || !state.runtime) return;
  const maxTokens = Number($('ws-length').value);
  if (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > 256) { status('Answer length must be between 2 and 256 tokens.'); return; }
  let observation;
  try {
    observation = observationFor(captures);
    if (!state.runtime?.policies.some(policy => policy.id === observation.policyId)) throw new Error('The selected capture is unavailable in this runtime.');
  } catch (error) { status(error.message); return; }
  if (!leavePartial()) return;
  state.partialSaved = false;
  state.partialDownload = false;
  state.abort = new AbortController();
  state.live = { prompt, tokens: [], stopped: false };
  setBusy(true);
  selection.select({ runId: null, position: null });
  render();
  try {
    const handle = await ensureModel($('ws-model').value, state.abort.signal);
    state.abort.signal.throwIfAborted();
    status('Thinking…');
    const receipt = await handle.inspect.generate(prompt, {
      ...observation,
      topKSize: 8,
      // Deterministic, so the same prompt gives the same run and two runs compare.
      generation: { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, presencePenalty: 0, maxTokens, useChatTemplate: true, signal: state.abort.signal },
      onEvent(event) {
        if (event.type !== 'token' || !state.live || state.abort?.signal.aborted) return;
        const record = event.token ?? null;
        state.live.tokens.push({ index: event.index, tokenId: event.tokenId, text: record?.text ?? null, probability: record?.probability ?? null, entropy: record?.entropy ?? null,
          candidates: record?.topCandidates?.map((c) => ({ tokenId: c.tokenId, text: c.text, probability: c.probability })) ?? null });
        scheduleRender();
      },
    });
    state.abort.signal.throwIfAborted();
    const run = await createRun({ receipt, prompt, environment: environment() });
    state.abort.signal.throwIfAborted();
    addRun(run);
    state.live = null;
    status('');
  } catch (error) {
    if (state.abort?.signal.aborted && state.live) {
      // What streamed stays on screen; without the finished receipt it is not a run.
      state.live.stopped = true;
      status(`Stopped after ${state.live.tokens.length} tokens.`);
    } else {
      if (state.live?.tokens.length) state.live.stopped = true;
      else state.live = null;
      status(`Could not answer: ${error.message}`);
    }
  }
  state.abort = null;
  setBusy(false);
}

function stop() {
  state.abort?.abort();
  status('Stopping…');
  updateSend();
}

const menuCaptures = () => ({ layers: $('ws-layers').checked, timing: $('ws-timing').checked, witnesses: $('ws-witnesses').checked });

// Every control that can start work reads state.busy while rendering, so a change re-renders.
function updateSend() {
  const button = $('ws-send');
  const id = $('ws-model').value;
  const selected = state.runtime?.details.find(model => model.modelId === id);
  const ready = state.model?.modelId === id;
  const download = selected?.sizeBytes ? `Download ${Math.ceil(selected.sizeBytes / 1e6)} MB & answer` : 'Download model & answer';
  const idle = !selected || ready ? 'Send' : state.runtime.stored.has(id) ? 'Load & answer' : download;
  button.textContent = state.abort?.signal.aborted ? 'Stopping…' : state.abort ? 'Stop' : state.busy ? 'Working…' : idle;
  button.disabled = state.importing ? true : state.busy ? !state.abort || state.abort.signal.aborted : !selected;
  button.classList.toggle('ws-stop', Boolean(state.abort));
}

function setBusy(busy) {
  state.busy = busy;
  updateSend();
  render();
}

let pending = false;
function scheduleRender() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}

// Opens on the answer's least likely token: a moment worth looking at.
function addRun(run, { select = true, saved = false, replaces = null } = {}) {
  if (!state.runs.some(existing => existing.id === run.id)) {
    retain(state.retained, run.id, run, replaces);
    state.runs = [run, ...state.runs.filter(item => item.id !== replaces)];
  }
  if (saved) state.saved.add(run.id);
  if (select) {
    const known = run.tokens.filter((token) => Number.isFinite(token.probability));
    const least = known.length ? known.reduce((low, token) => (token.probability < low.probability ? token : low)) : null;
    selection.select({ runId: run.id, position: least ? sequencePosition(run, least.index) : null });
  }
  render();
}

async function branch(run, index, tokenId) {
  // One model operation at a time: a second fork would start a second model load.
  if (state.busy || state.importing) return;
  let request;
  try { request = forkRequest(run, index, tokenId); } catch (error) { $('token-note').textContent = error.message; return; }
  state.abort = new AbortController();
  const signal = state.abort.signal;
  setBusy(true);
  status('Continuing from the chosen token…');
  try {
    const handle = await ensureModel(run.model.modelId, signal);
    signal.throwIfAborted();
    assertRunModel(run, handle.manifestHash);
    const chosen = run.observation.captureSelection;
    const receipt = await handle.inspect.generate(run.prompt, {
      policyId: run.observation.policyId,
      ...(chosen ? { capture: { roles: chosen.roles, layers: chosen.layers, tokens: chosen.tokens } } : {}),
      topKSize: 8,
      inputTokenIds: request.inputTokenIds,
      generation: { ...request.generation, signal },
    });
    const child = await createRun({ receipt, prompt: run.prompt, parent: request.parent, environment: environment() });
    signal.throwIfAborted();
    if (child.model.manifestHash !== run.model.manifestHash) throw new Error('the model files loaded now are not the ones this run used');
    addRun(child, { select: false });
    state.view = 'forest';
    status(`Branch ${child.id.slice(4, 10)}: ${child.tokens.length} tokens. It is under Branches.`);
  } catch (error) {
    status(signal.aborted ? 'Branch stopped.' : `Could not continue: ${error.message}`);
  }
  state.abort = null;
  setBusy(false);
  render();
}

// The same prompt again, adding what the run lacked (layer outputs or GPU time).
function recordAgain(run, { timing = false, layers = false }) {
  if (!state.runtime) { status('Recording needs WebGPU, which this browser does not provide.'); return; }
  if (!run.prompt) { status('This run did not keep its prompt, so it cannot be recorded again.'); return; }
  const option = [...$('ws-model').options].find((item) => item.value === run.model.modelId);
  if (option) $('ws-model').value = option.value;
  return ask(run.prompt, { layers: layers || Boolean(run.layerOutputs), timing: timing || Boolean(run.timing.gpu), witnesses: false });
}

function projectUnavailable(run) {
  if (state.busy) return 'A run is in progress.';
  if (!state.runtime) return 'Projecting needs the model on a GPU, which this browser does not provide.';
  if (!run.model.modelId) return 'This run did not keep which model produced it.';
  return '';
}

async function project(run, index) {
  if (state.busy || state.importing) return;
  const key = `${run.id}:${index}`;
  const chosen = choiceVectors(run, index);
  if (!chosen) return;
  state.projections.set(key, { state: 'running' });
  state.abort = new AbortController();
  const signal = state.abort.signal;
  setBusy(true);
  render();
  try {
    const handle = await ensureModel(run.model.modelId, signal);
    signal.throwIfAborted();
    assertRunModel(run, handle.manifestHash);
    if (typeof handle.inspect.projectLayerOutputs !== 'function') throw new Error('this Doppler runtime cannot project layer outputs');
    const result = await handle.inspect.projectLayerOutputs({ vectors: chosen.vectors.map((vector) => Float32Array.from(vector)), topK: 3 });
    signal.throwIfAborted();
    validateProjection(result, chosen.vectors.length);
    state.projections.set(key, { state: 'done', rows: result.rows });
  } catch (error) {
    state.projections.set(key, { state: 'failed', reason: signal.aborted ? 'Projection stopped.' : `Could not project: ${error.message}.` });
  }
  state.abort = null;
  setBusy(false);
  render();
}

// ---------------------------------------------------------------- examples

// The demo's recorded answer: tokens and alternatives, no inside-the-model capture.
async function openExample() {
  const { SAMPLE_INSPECTION_RECEIPT: receipt } = await import(`${DEMO}/data/sample-inspection.js`);
  const run = await createRun({ receipt, prompt: receipt.recording?.prompt ?? receipt.prompt ?? '', kind: 'recorded-example',
    environment: { runtimeVersion: receipt.fingerprint?.identity?.runtime?.version ?? null, runtimeChannel: 'recording', userAgent: null } });
  addRun(run, { saved: true });
  state.exampleRun = run.id;
  return receipt;
}

// The same answer with its recorded activity: layer outputs, attention queries and keys, memory.
async function openExampleActivity() {
  if (state.busy || state.importing) return;
  setBusy(true);
  state.abort = new AbortController();
  const signal = state.abort.signal;
  updateSend();
  status('Opening the recorded activity · 23 MB…');
  try {
    const { SAMPLE_INSPECTION_RECEIPT: receipt } = await import(`${DEMO}/data/sample-inspection.js`);
    const metadata = await (await fetch(`${DEMO}/assets/activity-example.json`, { signal })).json();
    const response = await fetch(`${DEMO}/assets/activity-example.json.gz`, { signal });
    if (!response.ok) throw new Error(`Activity file returned HTTP ${response.status}.`);
    const blob = await readRecordingBlob(response, metadata.compressedBytes);
    const bytes = await blob.arrayBuffer();
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((n) => n.toString(16).padStart(2, '0')).join('');
    if (hash !== metadata.sha256) throw new Error('the recording failed its integrity check');
    const { value, byteLength } = await readRecordingJson(blob, { signal });
    if (byteLength !== metadata.decodedBytes) throw new Error('Incorrect expanded activity size.');
    const { decodeRecording } = await import(`${DEMO}/ui/activation-grid/recording-codec.js`);
    const recording = decodeRecording(value);
    const same = recording.generatedTokenIds?.length === receipt.generatedTokenIds.length && recording.generatedTokenIds.every((id, i) => id === receipt.generatedTokenIds[i]);
    if (!same) throw new Error('the recorded activity belongs to a different answer');
    const run = await createRun({ receipt: { ...receipt, activationTrace: recording.trace }, prompt: metadata.prompt ?? receipt.recording?.prompt ?? '', kind: 'recorded-example',
      environment: { runtimeVersion: receipt.fingerprint?.identity?.runtime?.version ?? null, runtimeChannel: 'recording', userAgent: null } });
    signal.throwIfAborted();
    const position = selection.get().position;
    addRun(run, { select: false, saved: true, replaces: state.exampleRun });
    state.saved.delete(state.exampleRun);
    state.exampleRun = run.id;
    selection.select({ runId: run.id, position });
    status('');
  } catch (error) {
    status(signal.aborted ? 'Opening cancelled.' : `Could not open the recorded activity: ${error.message}`);
  }
  state.abort = null;
  setBusy(false);
  render();
}

// ---------------------------------------------------------------- rendering

function renderExamples() {
  const live = Boolean(state.runtime);
  $('ws-examples').replaceChildren(
    h('button', { type: 'button', className: 'ws-chip', disabled: state.busy, on: { click: () => (state.exampleRun ? showRun(state.exampleRun) : openExample()) } }, 'Recorded: why the sky is blue'),
    ...EXAMPLE_PROMPTS.slice(1).map((prompt) => h('button', { type: 'button', className: 'ws-chip', disabled: !live || state.busy, title: live ? '' : 'Needs WebGPU to answer',
      on: { click: () => { $('ws-prompt').value = prompt; $('ws-prompt').focus(); } } }, prompt)),
  );
}

// The answer's tokens, coloured by surprise. Selecting one selects it everywhere.
function renderAnswer(run, tokens, display) {
  const picked = run ? pickedIndex(run, display) : null;
  const known = (token) => (token && Number.isFinite(token.probability) ? token.probability : null);
  $('ws-prompt-echo').textContent = run?.prompt ?? state.live?.prompt ?? '';
  $('ws-tokens').replaceChildren(...tokens.map((token, i) => {
    const p = known(token);
    const near = (offset) => known(tokens[i + offset]) ?? p ?? 0.5;
    const text = token.text ?? `[${token.tokenId}]`;
    const space = !text.trim();
    return h('button', {
      type: 'button',
      className: ['token-chip', space ? 'token-chip--space' : '', p === null ? 'token-chip--unavailable' : '', i === picked ? 'is-selected' : '',
        token.inherited && !token.forked ? 'ws-inherited' : '', token.forked ? 'ws-forked' : ''].filter(Boolean).join(' '),
      ariaPressed: String(i === picked),
      disabled: !run,
      title: `${token.forked ? 'The token this branch put here' : token.inherited ? 'Inherited from the run this branched from' : `Token ${i + 1}`}${p === null ? '' : ` · ${percent(p)} likely`}`,
      style: `--token-surprisal: ${p === null ? 0.5 : 1 - p}; --token-gradient-start: ${1 - (near(-1) + (p ?? 0.5)) / 2}; --token-gradient-end: ${1 - (near(1) + (p ?? 0.5)) / 2}`,
      data: { index: i },
      on: { click: () => run && selection.select({ position: display.base === null ? sequencePosition(run, i) : display.base + i }), mouseenter: () => highlight(i), mouseleave: () => highlight(null) },
    }, space && text.includes('\n') ? `↵${text}` : text);
  }));
  const recorded = run?.kind === 'recorded-example';
  if (state.live?.stopped) { $('ws-answer-note').textContent = 'Stopped. You can save the partial answer as a transcript.'; return; }
  $('ws-answer-note').textContent = !tokens.length ? ''
    : display?.parent ? `Branch of run ${display.parent.id.slice(4, 10)}: ${display.inherited === 2 ? 'token 1 is' : `tokens 1–${display.inherited - 1} are`} inherited from it (lighter), token ${display.inherited} is the one this branch put there (outlined), and the rest it generated. Underline: blue where the model was sure, red where it was surprised.`
    : recorded ? `Recorded answer from ${run.model.modelId ?? 'a model'}. Underline: blue where the model was sure, red where it was surprised.`
      : 'Underline: blue where the model was sure, red where it was surprised. Select a token to look inside.';
}

function highlight(index) {
  for (const node of document.querySelectorAll('[data-index].is-hovered')) node.classList.remove('is-hovered');
  if (index === null) return;
  for (const node of document.querySelectorAll(`[data-index="${index}"]`)) node.classList.add('is-hovered');
}

// One column per answer token, in three aligned rows.
function renderSignals(run, tokens, display) {
  const rows = answerSignals(tokens);
  const picked = run ? pickedIndex(run, display) : null;
  const ppl = perplexity(tokens);
  $('ws-perplexity').replaceChildren(...(ppl
    ? [h('strong', {}, ppl.value.toFixed(2)), ppl.tokens === tokens.length
      ? ` perplexity over all ${tokens.length} answer tokens`
      : ` perplexity over the ${ppl.tokens} of ${tokens.length} answer tokens that recorded a probability`]
    : [tokens.length ? 'Perplexity: no probabilities recorded' : '']));
  if (!tokens.length) { $('ws-strip').replaceChildren(h('p', { className: 'demo-note' }, 'Ask something, or open the recorded example.')); return; }
  const maxSurprise = Math.max(1, ...rows.map((row) => row.surprise ?? 0));
  const maxConfusion = Math.max(1, ...rows.map((row) => row.confusion ?? 0));
  const row = (label, note, value, describe, variant = '') => h('div', { className: 'ws-row' },
    h('div', { className: 'ws-row-label' }, h('span', {}, label), h('span', { className: 'demo-note' }, note)),
    h('div', { className: `ws-bars ${variant}`, style: `--ws-columns: ${tokens.length}` }, ...rows.map((signal, i) => {
      const v = value(signal);
      return h('button', {
        type: 'button',
        className: ['ws-bar', v === null ? 'is-missing' : '', i === picked ? 'is-selected' : ''].filter(Boolean).join(' '),
        style: v === null ? '' : `--ws-value: ${Math.max(0.02, Math.min(1, v)).toFixed(3)}`,
        title: `Token ${i + 1} "${visible(tokens[i].text)}": ${describe(signal)}`,
        disabled: !run,
        data: { index: i },
        on: { click: () => run && selection.select({ position: display.base === null ? sequencePosition(run, i) : display.base + i }), mouseenter: () => highlight(i), mouseleave: () => highlight(null) },
      });
    })));
  $('ws-strip').replaceChildren(
    row('Surprise', '−ln p of the chosen token', (s) => (s.surprise === null ? null : s.surprise / maxSurprise),
      (s) => (s.surprise === null ? 'not recorded' : `${s.surprise.toFixed(2)} nats`)),
    // Full-distribution entropy when the runtime recorded it; otherwise the labelled top-candidate spread.
    rows.some((signal) => signal.confusion !== null)
      ? row('Confusion', 'entropy of the full next-token distribution', (s) => (s.confusion === null ? null : s.confusion / maxConfusion),
        (s) => (s.confusion === null ? 'not recorded' : `${s.confusion.toFixed(2)} nats`))
      : row('Spread', 'among the recorded top candidates, renormalized; not full entropy', (s) => (s.spread ? s.spread.entropy / s.spread.max : null),
        (s) => (s.spread ? `${s.spread.entropy.toFixed(2)} nats over ${s.spread.count} candidates holding ${percent(s.spread.coverage)}` : 'candidates not recorded')),
    row('Repetition', `share of repeated tokens in the last ${DIVERSITY_WINDOW}; a property of the text`, (s) => 1 - s.diversity.distinct,
      (s) => `${Math.round(s.diversity.distinct * s.diversity.window)} of ${s.diversity.window} distinct${s.diversity.repeated ? ', repeats an earlier token' : ''}`, 'ws-bars--neutral'),
  );
}

function renderTabs(run, display) {
  const recorded = new Set();
  if (run.tokens.some(token => token.candidates?.length)) recorded.add('atlas');
  if (run.parent || state.runs.some(child => child.parent?.runId === run.id)) recorded.add('forest');
  if (run.layerOutputs) { recorded.add('activity'); recorded.add('journey'); }
  // A bundled example can load its additional activity only on request.
  if (run.kind === 'recorded-example') recorded.add('activity');
  const activity = (run.activityRecording?.trace ?? run.activityTrace)?.activity;
  const roles = new Set(activity?.frames?.filter(frame => frame?.status === 'captured').map(frame => frame.role));
  if (roles.has('attention-query') && roles.has('kv-key')) recorded.add('attention');
  if (['kv-key', 'kv-value', 'recurrent-memory', 'convolution-history'].some(role => roles.has(role))) recorded.add('memory');
  if (roles.has('ffn-down-input') && activity.weights?.some(weight => weight?.status === 'captured' && ['down', 'ffnDown'].includes(weight.role))) recorded.add('gearbox');
  if (run.timing.gpu?.steps?.length) recorded.add('time');
  const views = state.allViews ? VIEWS : VIEWS.filter(view => recorded.has(view.id));
  if (!views.some(view => view.id === state.view)) state.view = views[0]?.id ?? null;
  $('ws-more-views').checked = state.allViews;
  $('ws-tabs').replaceChildren(...views.map((view) => h('button', {
    type: 'button', role: 'tab', className: 'ws-tab', id: `ws-tab-${view.id}`,
    tabIndex: view.id === state.view ? 0 : -1, ariaSelected: String(view.id === state.view), attrs: { 'aria-controls': `ws-view-${view.id}` },
    on: { click: () => { state.view = view.id; render(); } },
  }, view.label)));
  for (const view of VIEWS) {
    const panel = $(`ws-view-${view.id}`);
    panel.hidden = view.id !== state.view;
    panel.setAttribute('aria-label', view.label);
  }
  $('ws-view-picker').replaceChildren(...views.map(view => h('option', { value: view.id }, view.label)));
  $('ws-view-picker').value = state.view;
  const chosen = selection.get();
  $('ws-inside-model').textContent = run ? `${run.kind === 'recorded-example' ? 'Recorded example' : 'Inspecting answer'} · ${run.model.modelId ?? 'unknown model'}` : '';
  const index = run ? pickedIndex(run, display) : null;
  // Selection is available beside the views, including on a narrow screen.
  const words = run && display.base !== null ? display.tokens : [];
  const promptSelected = chosen.position !== null && index === null;
  $('ws-word').replaceChildren(h('option', { value: '' }, promptSelected ? `Prompt position ${chosen.position}` : 'Choose a token'),
    ...words.map((token, i) => h('option', { value: String(display.base + i) }, `${i + 1}: ${visible(token.text)}`)));
  $('ws-word').value = index === null ? '' : String(chosen.position);
  $('ws-word').disabled = words.length === 0;
  // One layer picker for every view; the journey and attention views can also set it.
  const layers = run ? (run.layerOutputs?.layers ?? run.activityTrace?.geometry?.layers ?? 0) : 0;
  $('ws-layer').replaceChildren(h('option', { value: '' }, 'All'), ...Array.from({ length: layers }, (_, i) => h('option', { value: String(i) }, String(i + 1))));
  $('ws-layer').value = chosen.layer === null || chosen.layer >= layers ? '' : String(chosen.layer);
  $('ws-layer').disabled = !layers;
  $('ws-selection').textContent = run ? `${index === null ? 'No token selected' : `Token ${index + 1}: "${visible(display.tokens[index].text)}"`} · ${chosen.layer === null ? 'all layers' : `layer ${chosen.layer + 1}`}` : '';
}

function renderInside(run, display) {
  document.querySelector('.ws-inside').hidden = !run;
  if (!run) return;
  const focusOnTab = document.activeElement?.getAttribute('role') === 'tab';
  renderTabs(run, display);
  if (focusOnTab) $(`ws-tab-${state.view}`)?.focus({ preventScroll: true });
  const holder = state.view ? $(`ws-view-${state.view}`) : null;
  const canRun = Boolean(state.runtime) && !state.busy;
  const why = state.runtime ? '' : 'Running the model needs WebGPU, which this browser does not provide.';
  const context = { run, runs: state.runs, selection, canRun, why, projections: state.projections, actions: { showRun, branch, recordAgain, project, projectUnavailable } };
  // The selected word's panel. An inherited word is shown as its own run recorded it,
  // so its candidates, branches and actions are the same as on that run.
  const k = pickedIndex(run, display), word = k === null ? null : display.tokens[k];
  if (word && word.owner !== run) {
    const owner = word.owner, position = sequencePosition(owner, word.ownIndex);
    renderTokenPanel({ ...context, run: owner, selection: { ...selection, get: () => ({ ...selection.get(), runId: owner.id, position }) },
      note: word.forked ? ` This branch put "${visible(word.text)}" here instead; these are the choices run ${owner.id.slice(4, 10)} had.` : ` Inherited from run ${owner.id.slice(4, 10)}.` });
  } else {
    renderTokenPanel(context);
  }
  renderForestTree(context);
  const view = VIEWS.find((item) => item.id === state.view);
  view?.render?.(context);
  // A recorded example without its activity offers to open it here.
  // Only the visible view carries the offer; a hidden one would keep a stale copy.
  for (const stale of document.querySelectorAll('.ws-load-activity')) stale.remove();
  if (run.kind === 'recorded-example' && !run.layerOutputs && ['activity', 'journey', 'attention', 'memory', 'gearbox'].includes(state.view)) {
    holder.prepend(h('div', { className: 'ws-load-activity status-row' }, h('span', {}, 'This recorded answer\'s inside-the-model activity is a separate 23 MB file.'),
      h('button', { type: 'button', className: 'btn btn-small', disabled: state.busy, on: { click: openExampleActivity } }, 'Open it')));
  }
}

function unsaved(run) { return !state.saved.has(run.id); }
function warnBeforeUnload(event) { event.preventDefault(); event.returnValue = ''; }
function updateUnloadWarning() {
  const pending = state.runs.some(unsaved) || (state.live?.tokens.length && !state.partialSaved);
  if (pending) window.addEventListener('beforeunload', warnBeforeUnload);
  else window.removeEventListener('beforeunload', warnBeforeUnload);
}

function renderRuns() {
  const current = selection.get().runId;
  $('ws-clear').hidden = !state.runs.length;
  $('ws-clear').disabled = state.busy || Boolean(state.importing);
  $('ws-budget').textContent = state.runs.length ? `Estimated memory: ${Math.ceil(retainedBytes(state.retained) / 1048576)} of ${retainedLimit / 1048576} MiB` : '';
  $('ws-save-partial').hidden = !state.live?.stopped || !state.live.tokens.length;
  $('ws-mark-partial').hidden = !state.live?.stopped || !state.partialDownload || state.partialSaved;
  $('ws-cancel-import').hidden = !state.importing;
  updateSend();
  $('ws-open-button').disabled = state.busy || Boolean(state.importing);
  $('ws-runs').replaceChildren(...state.runs.map((run) => h('li', { className: run.id === current ? 'is-current' : '' },
    h('button', { type: 'button', className: 'ws-run', ariaPressed: String(run.id === current), disabled: state.busy, on: { click: () => showRun(run.id) } },
      h('code', {}, run.id.slice(4, 10)), ' ',
      h('span', {}, (run.prompt || '(no prompt)').slice(0, 70)),
      h('span', { className: 'demo-note' }, ` · ${run.tokens.length} tokens${unsaved(run) ? ' · Unsaved' : ''}${run.parent ? ` · branch at token ${run.parent.index + 1}` : ''}${run.layerOutputs ? ' · layers' : ''}${run.activityTrace ? ' · activity' : ''}`)),
    h('button', { type: 'button', className: 'btn btn-small btn-ghost', on: { click: () => save(run) } }, 'Save'),
    state.downloads.has(run.id) && unsaved(run) ? h('button', { type: 'button', className: 'btn btn-small btn-ghost', on: { click: () => { state.saved.add(run.id); renderRuns(); } } }, 'Mark saved') : null,
    h('button', { type: 'button', className: 'btn btn-small btn-ghost', disabled: state.busy || Boolean(state.importing), on: { click: () => removeRuns([run]) } }, 'Remove'))));
  $('ws-summary').hidden = !shown();
  updateUnloadWarning();
}

function removeRuns(list) {
  if (list.some(unsaved) && !confirm('Discard unsaved recordings?')) return;
  const removed = new Set(list.map(run => run.id));
  state.runs = state.runs.filter(run => !removed.has(run.id));
  for (const id of removed) { state.saved.delete(id); state.downloads.delete(id); state.retained.delete(id); }
  for (const key of state.projections.keys()) if (removed.has(key.split(':')[0])) state.projections.delete(key);
  if (removed.has(state.exampleRun)) state.exampleRun = null;
  const current = selection.get();
  selection.select({ runId: removed.has(current.runId) ? state.runs[0]?.id ?? null : current.runId,
    compareRunId: removed.has(current.compareRunId) ? null : current.compareRunId });
  render();
}

function save(run) {
  downloadFile(`doppler-${run.id}.json`, serializeRun(run));
  state.downloads.add(run.id);
  if (unsaved(run)) status('When your download finishes, choose Mark saved.');
  renderRuns();
}

async function openFiles(files) {
  if (state.importing || state.busy || !files.length || !leavePartial()) return;
  state.importing = new AbortController();
  const signal = state.importing.signal;
  const problems = [];
  render();
  try {
    for (const file of files) {
      status(`Opening ${file.name}…`);
      try {
        const { value } = await readRecordingJson(file, { signal });
        signal.throwIfAborted();
        addRun(validateRun(value), { saved: true });
      } catch (error) {
        if (signal.aborted) break;
        problems.push(`${file.name}: ${error.message}`);
      }
    }
    status([signal.aborted ? 'Opening cancelled.' : '', ...problems].filter(Boolean).join(' '));
  } finally {
    state.importing = null;
    $('ws-open').value = '';
    render();
  }
}

// The run picked to compare with the shown one; the faultline view does the comparing.
function renderCompare(run) {
  const others = run ? state.runs.filter((item) => item.id !== run.id) : [];
  const chosen = selection.get().compareRunId;
  $('ws-compare-with').replaceChildren(h('option', { value: '' }, others.length ? 'Choose a run' : 'No other run'),
    ...others.map((item) => h('option', { value: item.id }, `${item.id.slice(4, 10)} · ${(item.prompt || '(no prompt)').slice(0, 40)} · ${item.tokens.length} tokens${item.parent ? ' · branch' : ''}`)));
  $('ws-compare-with').value = others.some((item) => item.id === chosen) ? chosen : '';
  $('ws-compare-with').disabled = !others.length;
  const other = others.find((item) => item.id === chosen) ?? null;
  document.querySelector('.ws-compare').hidden = !others.length;
  $('ws-compare-empty').hidden = Boolean(other);
  renderFaultline({ run: other ? run : null, other, selection });
}

function render() {
  const run = state.live ? null : shown();
  const display = run ? displayOf(run) : { tokens: state.live?.tokens ?? [], inherited: 0, base: null, parent: null };
  const tokens = display.tokens;
  renderExamples();
  renderAnswer(run, tokens, display);
  renderSignals(run, tokens, display);
  $('ws-token-empty').hidden = !run || pickedIndex(run, display) !== null;
  document.querySelector('.ws-signals').hidden = !tokens.length;
  if (!run) $('token-panel').hidden = true;
  renderInside(run, display);
  renderRuns();
  renderCompare(run);
}

selection.subscribe(render);

// ---------------------------------------------------------------------- boot

async function boot() {
  const flags = await loadFlags().catch(() => ({}));
  mountNav('explore', flags);
  $('ws-form').addEventListener('submit', (event) => { event.preventDefault(); if (state.abort) stop(); else ask($('ws-prompt').value.trim()); });
  $('ws-prompt').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (!state.busy) ask($('ws-prompt').value.trim()); }
  });
  $('ws-model').addEventListener('change', syncMenuLabel);
  $('ws-more-views').addEventListener('change', event => { state.allViews = event.target.checked; render(); });
  $('ws-tabs').addEventListener('keydown', event => {
    const tabs = [...$('ws-tabs').querySelectorAll('[role=tab]')];
    const i = tabs.indexOf(event.target);
    if (i < 0) return;
    const next = { ArrowLeft: (i - 1 + tabs.length) % tabs.length, ArrowRight: (i + 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    state.view = tabs[next].id.slice('ws-tab-'.length); render(); $(`ws-tab-${state.view}`).focus();
  });
  $('ws-clear').addEventListener('click', () => removeRuns(state.runs));
  $('ws-cancel-import').addEventListener('click', () => state.importing?.abort());
  $('ws-summary').addEventListener('click', () => {
    if (shown()) downloadFile('doppler-redacted-summary.json', JSON.stringify(redactedSummary(shown()), null, 2));
  });
  $('ws-save-partial').addEventListener('click', () => {
    if (!state.live?.stopped) return;
    downloadFile('doppler-partial-answer.txt', state.live.tokens.map(token => token.text ?? '').join(''), 'text/plain');
    state.partialDownload = true; renderRuns(); status('When your text download finishes, choose Mark saved.');
  });
  $('ws-mark-partial').addEventListener('click', () => {
    state.partialSaved = true; renderRuns(); status('Partial answer marked saved.');
  });
  $('ws-view-picker').addEventListener('change', (event) => { state.view = event.target.value; render(); });
  $('ws-word').addEventListener('change', (event) => selection.select({ position: event.target.value === '' ? null : Number(event.target.value) }));
  $('ws-layer').addEventListener('change', (event) => selection.select({ layer: event.target.value === '' ? null : Number(event.target.value) }));
  $('ws-open-button').addEventListener('click', () => $('ws-open').click());
  $('ws-open').addEventListener('change', (event) => openFiles([...event.target.files]));
  $('ws-compare-with').addEventListener('change', (event) => selection.select({ compareRunId: event.target.value || null }));
  $('tolerance').addEventListener('input', render);
  globalThis.__DOPPLER_WORKSPACE__ = state;
  render();
  await openExample().catch((error) => status(`The recorded example did not open: ${error.message}`));
  // An answer handed over from the Benchmark page opens as a run, like a saved one.
  const handed = take('run');
  if (handed) {
    try { addRun(validateRun(handed)); status('Opened the answer from the benchmark. It has no layer outputs: the benchmark records only tokens and time.'); }
    catch (error) { status(error.message); }
  }
  registerOfflineWorker();

  state.device = await describeDevice();
  if (state.device.webgpu !== 'available') {
    $('ws-model').replaceChildren(h('option', {}, 'Needs WebGPU'));
    $('ws-runtime').textContent = 'This browser can’t run models here: WebGPU isn’t available. Recorded answers and saved runs still open.';
    $('ws-prompt').placeholder = 'This browser can’t run models. Explore the recorded answer below.';
    render();
    return;
  }
  status('Loading Doppler…');
  try {
    state.choice = await loadRuntimeChoice(flags);
    $('ws-runtime').textContent = `Runtime: ${describeRuntime(state.choice)}`;
    await loadRuntime();
    setBusy(false);
    status('');
  } catch (error) {
    status(`Doppler did not load, so this page cannot answer: ${error.message}`);
  }
  render();
}

// Keeps the page's files for offline use; models are saved by the runtime itself.
async function registerOfflineWorker() {
  if (!globalThis.isSecureContext || !('serviceWorker' in navigator) || state.choice?.preview || new URLSearchParams(location.search).get('runtime') === 'preview') return;
  try { await navigator.serviceWorker.register(new URL('./sw.js', import.meta.url), { scope: './' }); } catch { /* the page works online without it */ }
}

boot();
