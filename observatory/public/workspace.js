// The workspace: one prompt, its answer streaming below, and every view of the
// run following one shared selection. Recording needs WebGPU; recorded runs and
// saved files open without it.

import { loadFlags } from './flags.js';
import { describeDevice } from './run.js';
import { mountNav, readSavedText } from './site.js';
import { take } from './handoff.js';
import { describeRuntime, loadRuntimeChoice } from './runtimes.js';
import { answerIndex, choiceVectors, createRun, forkRequest, sequencePosition, serializeRun, validateRun } from './shared-run.js';
import { createSelection } from './selection.js';
import { $, h, percent, visible } from './viz/dom.js';
import { answerSignals, DIVERSITY_WINDOW, perplexity } from './viz/signals.js';
import { renderForestTree, renderTokenPanel } from './views/forest.js';
import { renderJourney } from './views/journey.js';
import { renderAttention } from './views/attention.js';
import { renderMemory } from './views/memory.js';
import { renderGearbox } from './views/gearbox.js';
import { renderAtlas } from './views/atlas.js';
import { renderConcertina } from './views/concertina.js';
import { renderFaultline } from './views/faultline.js';
import { assertRunModel, validateProjection } from './projection-contract.js';

const DEMO = '/doppler/demo';
const EXAMPLE_PROMPTS = ['Explain why the sky is blue in one sentence.', 'Write a haiku about rain.', 'What is 17 × 23?', 'Name three uses of a paperclip.'];
const VIEWS = [
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
  view: 'journey',
  device: null,
  choice: null,
  runtime: null, // { dr, version, commit, channel, selection }
  model: null,
  busy: false,
  abort: null, // stops the answer being generated
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
const showRun = (runId) => { if (state.live?.stopped) state.live = null; selection.select({ runId }); render(); };
const status = (text) => { $('ws-status').textContent = text; };

// ------------------------------------------------------------------ runtime

async function loadRuntime() {
  const module = await import(state.choice.moduleUrl);
  const dr = module.dr ?? module.doppler;
  const details = (await dr.listModelDetails()).filter((entry) => entry?.modes?.includes('text') && entry?.classification?.tasks?.includes('generation'));
  let stored = new Set();
  try { stored = new Set((await dr.listPersistentModels()).map((entry) => entry.modelId)); } catch { /* shown as not saved */ }
  $('ws-model').replaceChildren(...details.map((entry) => h('option', { value: entry.modelId },
    `${entry.label ?? entry.modelId}${stored.has(entry.modelId) ? ' · saved' : entry.sizeBytes ? ` · ${(entry.sizeBytes / 1e6).toFixed(0)} MB` : ''}`)));
  const saved = details.find((entry) => stored.has(entry.modelId));
  if (saved) $('ws-model').value = saved.modelId;
  $('ws-model').disabled = false;
  state.runtime = { dr, version: module.DOPPLER_VERSION ?? null, commit: state.choice.release?.sourceCommit ?? null, channel: state.choice.id, selection: false };
  syncMenuLabel();
}

function syncMenuLabel() {
  const option = $('ws-model').selectedOptions[0];
  $('ws-menu-label').textContent = state.runtime && option ? option.textContent.split(' · ')[0] : 'Model';
}

async function ensureModel(modelId) {
  if (state.model?.modelId === modelId) return state.model.handle;
  if (state.model) { await state.model.handle.unload?.().catch(() => {}); state.model = null; }
  const handle = await state.runtime.dr.load(modelId, { cache: 'opfs', onProgress: (event) => status(event?.message ?? 'Loading the model…') });
  try { state.runtime.selection = handle.inspect.listPolicies().some((policy) => policy.activationTrace?.activitySelection !== undefined); } catch { state.runtime.selection = false; }
  state.model = { modelId, handle };
  return handle;
}

// Which observation tier records what the menu asks for.
function observationFor({ layers, timing, witnesses }) {
  if (witnesses) {
    if (!state.runtime.selection) throw new Error('this runtime cannot record attention, memory and weights');
    return { policyId: 'demo/model-activity-timing' };
  }
  if (layers && timing) {
    if (!state.runtime.selection) throw new Error('this runtime cannot record layer outputs and GPU time in one run; choose one in the menu');
    return { policyId: 'demo/model-activity-timing', capture: { roles: [] } };
  }
  if (layers) return { policyId: 'demo/activation-grid' };
  if (timing) return { policyId: 'demo/deep-xray' };
  return { policyId: 'demo/guided-quality' };
}

const environment = () => ({ runtimeVersion: state.runtime.version, runtimeCommit: state.runtime.commit, runtimeChannel: state.runtime.channel, userAgent: navigator.userAgent });

// ------------------------------------------------------------------ asking

async function ask(prompt, captures = menuCaptures()) {
  if (!prompt || state.busy) return;
  const maxTokens = Number($('ws-length').value);
  if (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > 256) { status('Answer length must be between 2 and 256 tokens.'); return; }
  state.abort = new AbortController();
  state.live = { prompt, tokens: [], stopped: false };
  setBusy(true);
  selection.select({ runId: null, position: null });
  render();
  try {
    const handle = await ensureModel($('ws-model').value);
    status('Thinking…');
    const receipt = await handle.inspect.generate(prompt, {
      ...observationFor(captures),
      topKSize: 8,
      // Deterministic, so the same prompt gives the same run and two runs compare.
      generation: { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, presencePenalty: 0, maxTokens, useChatTemplate: true, signal: state.abort.signal },
      onEvent(event) {
        if (event.type !== 'token' || !state.live) return;
        const record = event.token ?? null;
        state.live.tokens.push({ index: event.index, tokenId: event.tokenId, text: record?.text ?? null, probability: record?.probability ?? null, entropy: record?.entropy ?? null,
          candidates: record?.topCandidates?.map((c) => ({ tokenId: c.tokenId, text: c.text, probability: c.probability })) ?? null });
        scheduleRender();
      },
    });
    const run = await createRun({ receipt, prompt, environment: environment() });
    state.live = null;
    addRun(run);
    status('');
  } catch (error) {
    if (state.abort?.signal.aborted && state.live) {
      // What streamed stays on screen; without the finished receipt it is not a run.
      state.live.stopped = true;
      status(`Stopped after ${state.live.tokens.length} tokens. A stopped answer is not kept as a run.`);
    } else {
      state.live = null;
      status(`Could not answer: ${error.message}`);
    }
  }
  state.abort = null;
  setBusy(false);
}

function stop() { state.abort?.abort(); }

const menuCaptures = () => ({ layers: $('ws-layers').checked, timing: $('ws-timing').checked, witnesses: $('ws-witnesses').checked });

// Every control that can start work reads state.busy while rendering, so a change re-renders.
function setBusy(busy) {
  state.busy = busy;
  // While an answer streams, Send becomes Stop; other work (a fork, a projection) cannot be stopped midway.
  const stoppable = busy && Boolean(state.abort);
  $('ws-send').disabled = !state.runtime || (busy && !stoppable);
  $('ws-send').textContent = stoppable ? 'Stop' : busy ? 'Working…' : 'Send';
  $('ws-send').classList.toggle('ws-stop', stoppable);
  render();
}

let pending = false;
function scheduleRender() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}

// Opens on the answer's least likely token: a moment worth looking at.
function addRun(run, { select = true } = {}) {
  if (state.live?.stopped) state.live = null;
  if (!state.runs.some((existing) => existing.id === run.id)) state.runs.unshift(run);
  if (select) {
    const known = run.tokens.filter((token) => Number.isFinite(token.probability));
    const least = known.length ? known.reduce((low, token) => (token.probability < low.probability ? token : low)) : null;
    selection.select({ runId: run.id, position: least ? sequencePosition(run, least.index) : null });
  }
  render();
}

async function branch(run, index, tokenId) {
  // One model operation at a time: a second fork would start a second model load.
  if (state.busy) return;
  let request;
  try { request = forkRequest(run, index, tokenId); } catch (error) { $('token-note').textContent = error.message; return; }
  setBusy(true);
  status('Continuing from the chosen token…');
  try {
    const handle = await ensureModel(run.model.modelId);
    assertRunModel(run, handle.manifestHash);
    const chosen = run.observation.captureSelection;
    const receipt = await handle.inspect.generate(run.prompt, {
      policyId: run.observation.policyId,
      ...(chosen ? { capture: { roles: chosen.roles, layers: chosen.layers, tokens: chosen.tokens } } : {}),
      topKSize: 8,
      inputTokenIds: request.inputTokenIds,
      generation: request.generation,
    });
    const child = await createRun({ receipt, prompt: run.prompt, parent: request.parent, environment: environment() });
    if (child.model.manifestHash !== run.model.manifestHash) throw new Error('the model files loaded now are not the ones this run used');
    addRun(child, { select: false });
    state.view = 'forest';
    status(`Branch ${child.id.slice(4, 10)}: ${child.tokens.length} tokens. It is under Branches.`);
  } catch (error) {
    status(`Could not continue: ${error.message}`);
  }
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
  if (state.busy) return;
  const key = `${run.id}:${index}`;
  const chosen = choiceVectors(run, index);
  if (!chosen) return;
  state.projections.set(key, { state: 'running' });
  setBusy(true);
  render();
  try {
    const handle = await ensureModel(run.model.modelId);
    assertRunModel(run, handle.manifestHash);
    if (typeof handle.inspect.projectLayerOutputs !== 'function') throw new Error('this Doppler runtime cannot project layer outputs');
    const result = await handle.inspect.projectLayerOutputs({ vectors: chosen.vectors.map((vector) => Float32Array.from(vector)), topK: 3 });
    validateProjection(result, chosen.vectors.length);
    state.projections.set(key, { state: 'done', rows: result.rows });
  } catch (error) {
    state.projections.set(key, { state: 'failed', reason: `Could not project: ${error.message}.` });
  }
  setBusy(false);
  render();
}

// ---------------------------------------------------------------- examples

// The demo's recorded answer: tokens and alternatives, no inside-the-model capture.
async function openExample() {
  const { SAMPLE_INSPECTION_RECEIPT: receipt } = await import(`${DEMO}/data/sample-inspection.js`);
  const run = await createRun({ receipt, prompt: receipt.recording?.prompt ?? receipt.prompt ?? '', kind: 'recorded-example',
    environment: { runtimeVersion: receipt.fingerprint?.identity?.runtime?.version ?? null, runtimeChannel: 'recording', userAgent: null } });
  state.exampleRun = run.id;
  addRun(run);
  return receipt;
}

// The same answer with its recorded activity: layer outputs, attention queries and keys, memory.
async function openExampleActivity() {
  if (state.busy) return;
  setBusy(true);
  status('Opening the recorded activity · 23 MB…');
  try {
    const { SAMPLE_INSPECTION_RECEIPT: receipt } = await import(`${DEMO}/data/sample-inspection.js`);
    const metadata = await (await fetch(`${DEMO}/assets/activity-example.json`)).json();
    const bytes = await (await fetch(`${DEMO}/assets/activity-example.json.gz`)).arrayBuffer();
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map((n) => n.toString(16).padStart(2, '0')).join('');
    if (hash !== metadata.sha256) throw new Error('the recording failed its integrity check');
    const text = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
    const { decodeRecording } = await import(`${DEMO}/ui/activation-grid/recording-codec.js`);
    const recording = decodeRecording(JSON.parse(text));
    const same = recording.generatedTokenIds?.length === receipt.generatedTokenIds.length && recording.generatedTokenIds.every((id, i) => id === receipt.generatedTokenIds[i]);
    if (!same) throw new Error('the recorded activity belongs to a different answer');
    const run = await createRun({ receipt: { ...receipt, activationTrace: recording.trace }, prompt: metadata.prompt ?? receipt.recording?.prompt ?? '', kind: 'recorded-example',
      environment: { runtimeVersion: receipt.fingerprint?.identity?.runtime?.version ?? null, runtimeChannel: 'recording', userAgent: null } });
    state.runs = state.runs.filter((item) => item.id !== state.exampleRun);
    state.exampleRun = run.id;
    const position = selection.get().position;
    addRun(run, { select: false });
    selection.select({ runId: run.id, position });
    status('');
  } catch (error) {
    status(`Could not open the recorded activity: ${error.message}`);
  }
  setBusy(false);
  render();
}

// ---------------------------------------------------------------- rendering

function renderExamples() {
  const live = Boolean(state.runtime);
  $('ws-examples').replaceChildren(
    h('button', { type: 'button', className: 'ws-chip', disabled: state.busy, on: { click: () => (state.exampleRun ? showRun(state.exampleRun) : openExample()) } }, 'Recorded: why the sky is blue'),
    ...EXAMPLE_PROMPTS.slice(1).map((prompt) => h('button', { type: 'button', className: 'ws-chip', disabled: !live || state.busy, title: live ? '' : 'Needs WebGPU to answer',
      on: { click: () => { $('ws-prompt').value = prompt; ask(prompt); } } }, prompt)),
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
      title: `${token.forked ? 'The word this branch put here' : token.inherited ? 'Inherited from the run this branched from' : `Token ${i + 1}`}${p === null ? '' : ` · ${percent(p)} likely`}`,
      style: `--token-surprisal: ${p === null ? 0.5 : 1 - p}; --token-gradient-start: ${1 - (near(-1) + (p ?? 0.5)) / 2}; --token-gradient-end: ${1 - (near(1) + (p ?? 0.5)) / 2}`,
      data: { index: i },
      on: { click: () => run && selection.select({ position: display.base === null ? sequencePosition(run, i) : display.base + i }), mouseenter: () => highlight(i), mouseleave: () => highlight(null) },
    }, space && text.includes('\n') ? `↵${text}` : text);
  }));
  const recorded = run?.kind === 'recorded-example';
  if (state.live?.stopped) { $('ws-answer-note').textContent = 'Stopped. This partial answer is not saved as a run.'; return; }
  $('ws-answer-note').textContent = !tokens.length ? ''
    : display?.parent ? `Branch of run ${display.parent.id.slice(4, 10)}: ${display.inherited === 2 ? 'word 1 is' : `words 1–${display.inherited - 1} are`} inherited from it (lighter), word ${display.inherited} is the one this branch put there (outlined), and the rest it generated. Underline: blue where the model was sure, red where it was surprised.`
    : recorded ? `Recorded answer from ${run.model.modelId ?? 'a model'}. Underline: blue where the model was sure, red where it was surprised.`
      : 'Underline: blue where the model was sure, red where it was surprised. Select a word to look inside.';
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
  $('ws-tabs').replaceChildren(...VIEWS.map((view) => h('button', {
    type: 'button', role: 'tab', className: 'ws-tab', id: `ws-tab-${view.id}`,
    ariaSelected: String(view.id === state.view), attrs: { 'aria-controls': `ws-view-${view.id}` },
    on: { click: () => { state.view = view.id; render(); } },
  }, view.label)));
  for (const view of VIEWS) $(`ws-view-${view.id}`).hidden = view.id !== state.view;
  const chosen = selection.get();
  const index = run ? pickedIndex(run, display) : null;
  // One layer picker for every view; the journey and attention views can also set it.
  const layers = run ? (run.layerOutputs?.layers ?? run.activityTrace?.geometry?.layers ?? 0) : 0;
  $('ws-layer').replaceChildren(h('option', { value: '' }, 'All'), ...Array.from({ length: layers }, (_, i) => h('option', { value: String(i) }, String(i + 1))));
  $('ws-layer').value = chosen.layer === null || chosen.layer >= layers ? '' : String(chosen.layer);
  $('ws-layer').disabled = !layers;
  $('ws-selection').textContent = run ? `${index === null ? 'No word selected' : `Word ${index + 1}: "${visible(display.tokens[index].text)}"`} · ${chosen.layer === null ? 'all layers' : `layer ${chosen.layer + 1}`}` : '';
}

function renderInside(run, display) {
  renderTabs(run, display);
  const holder = $(`ws-view-${state.view}`);
  if (!run) { return; }
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
  view.render?.(context);
  // A recorded example without its activity offers to open it here.
  holder.querySelector('.ws-load-activity')?.remove();
  if (run.kind === 'recorded-example' && !run.layerOutputs && ['journey', 'attention', 'memory', 'gearbox'].includes(state.view)) {
    holder.prepend(h('div', { className: 'ws-load-activity status-row' }, h('span', {}, 'This recorded answer\'s inside-the-model activity is a separate 23 MB file.'),
      h('button', { type: 'button', className: 'btn btn-small', disabled: state.busy, on: { click: openExampleActivity } }, 'Open it')));
  }
}

function renderRuns() {
  const current = selection.get().runId;
  $('ws-runs').replaceChildren(...state.runs.map((run) => h('li', { className: run.id === current ? 'is-current' : '' },
    h('button', { type: 'button', className: 'ws-run', ariaPressed: String(run.id === current), on: { click: () => showRun(run.id) } },
      h('code', {}, run.id.slice(4, 10)), ' ',
      h('span', {}, (run.prompt || '(no prompt)').slice(0, 70)),
      h('span', { className: 'demo-note' }, ` · ${run.tokens.length} tokens${run.parent ? ` · branch at word ${run.parent.index + 1}` : ''}${run.layerOutputs ? ' · layers' : ''}${run.activityTrace ? ' · activity' : ''}`)),
    h('button', { type: 'button', className: 'btn btn-small btn-ghost', on: { click: () => save(run) } }, 'Save'))));
}

function save(run) {
  const link = h('a', { href: URL.createObjectURL(new Blob([serializeRun(run)], { type: 'application/json' })), download: `doppler-${run.id}.json` });
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function openFiles(files) {
  const problems = [];
  for (const file of files) {
    try { addRun(validateRun(JSON.parse(await readSavedText(file)))); }
    catch (error) { problems.push(`${file.name}: ${error instanceof SyntaxError ? 'not JSON' : error.message}`); }
  }
  status(problems.join(' '));
  $('ws-open').value = '';
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
  $('ws-token-empty').hidden = Boolean(run && pickedIndex(run, display) !== null);
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
