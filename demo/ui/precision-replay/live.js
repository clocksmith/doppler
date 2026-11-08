import * as browserRuntime from 'doppler-gpu';
// The demo's browser import map resolves the supported host API; Node's root types expose the minimal Capsule API.
const { dr } = /** @type {typeof import('doppler-gpu/compat')} */ (/** @type {unknown} */ (browserRuntime));
import { loadRuntimeProfile } from 'doppler-gpu/tooling/runtime';
import { state } from '../state.js';
import { getPrompt, setGenerating } from '../../input.js';
import { setPhase } from '../../output.js';

/** @param {string} id */
const $ = id => document.getElementById(id);

/** @type {Record<string, unknown> | null} */
let comparison = null;

// The message box, or the last message sent when the box is empty. Earlier
// turns are not included: both runs answer this one prompt.
function comparisonPrompt() {
  const typed = getPrompt();
  if (typed) return typed;
  return [...state.conversationHistory].reverse().find((message) => message.role === 'user')?.content ?? '';
}

/** @param {string | null | undefined} dtype */
const precisionLabel = (dtype) => ({ f32: '32-bit', f16: '16-bit' })[String(dtype)] ?? (dtype || 'not reported');
const LANE_LABELS = { 'Default path': 'As usual', 'F16-requested path': '16-bit requested' };

export function initLivePrecision() {
  $('live-precision-run')?.addEventListener('click', () => { void runLivePrecision(); });
  $('live-precision-export')?.addEventListener('click', () => {
    if (!comparison) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(comparison)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'doppler-live-precision.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  });
}

export async function runLivePrecision() {
  const status = $('live-precision-status'), results = $('live-precision-results');
  if (!status || !results) return;
  const prompt = comparisonPrompt();
  if (!state.model || !state.modelId || !prompt) { status.textContent = 'Load a model and write a prompt first.'; return; }
  if (state.generating || state.modelBusy || state.settingsBusy) return;
  if (state.settings.runtimeProfile !== 'profiles/default') { status.textContent = 'Set Options → Engine profile to Default first, so both runs start from the same settings.'; return; }
  const model = /** @type {import('doppler-gpu/tooling/evidence').DopplerModelHandle} */ (state.model);
  const generation = { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, seed: 0, maxTokens: 16, useChatTemplate: true };
  state.generating = true;
  state.abortController = new AbortController();
  const signal = state.abortController.signal;
  setGenerating(true); setPhase('Comparing precision');
  status.textContent = '';
  results.replaceChildren();
  comparison = { schema: 'doppler.live-precision-comparison/v1', prompt, modelId: state.modelId,
    candidateProfile: 'profiles/demo-f16-comparison', generation, completed: false, lanes: [] };
  /** @type {import('doppler-gpu/tooling/evidence').DopplerModelHandle | null} */
  let candidate = null;
  const lanes = [];
  try {
    status.textContent = 'Running as usual…';
    const baseline = await model.inspect.generate(prompt, { policyId: 'demo/guided-quality', generation: { ...generation, signal } });
    lanes.push({ name: 'Default path', receipt: baseline });
    comparison.lanes = lanes;
    if (!baseline.inputTokenIds?.length) throw new Error('The default path did not record its exact prompt tokens.');
    signal.throwIfAborted();
    status.textContent = 'Loading a second copy that asks for 16-bit math…';
    const { runtime } = await loadRuntimeProfile('profiles/demo-f16-comparison');
    signal.throwIfAborted();
    const opened = await dr.load(state.modelId, { cache: 'opfs', runtimeConfig: runtime, isolatedLoader: true, signal });
    candidate = opened;
    if (opened.manifestHash !== model.manifestHash) throw new Error('The two copies loaded different model files.');
    signal.throwIfAborted();
    const observed = await opened.inspect.generate(prompt, { policyId: 'demo/guided-quality',
      inputTokenIds: baseline.inputTokenIds, generation: { ...generation, signal } });
    lanes.push({ name: 'F16-requested path', receipt: observed });
    if (JSON.stringify(observed.inputTokenIds) !== JSON.stringify(baseline.inputTokenIds)) throw new Error('The two runs read the prompt as different tokens.');
    const index = baseline.generatedTokenIds.findIndex((id, i) => observed.generatedTokenIds[i] !== id);
    const divergence = index >= 0 ? index : baseline.generatedTokenIds.length === observed.generatedTokenIds.length ? null : baseline.generatedTokenIds.length;
    comparison.completed = true; comparison.firstDifferentOutputIndex = divergence;
    status.textContent = divergence === null
      ? `Both runs wrote exactly the same ${baseline.generatedTokenIds.length} tokens.`
      : `The runs first differ at token ${divergence + 1}; after that, each continues from its own text.`;
  } catch (error) {
    comparison.failure = String(error);
    status.textContent = signal.aborted ? 'Stopped. Runs that finished are kept.'
      : `Could not compare: ${String(error)}. Not every model or device supports 16-bit math.`;
  } finally {
    try { await candidate?.unload(); }
    catch (error) { comparison.cleanupFailure = String(error); status.textContent += ` Closing the second copy failed: ${String(error)}`; }
    for (const lane of lanes) {
      const article = document.createElement('article'); article.className = 'precision-card';
      const title = document.createElement('div'); title.className = 'type-label';
      title.textContent = LANE_LABELS[/** @type {keyof typeof LANE_LABELS} */ (lane.name)] ?? lane.name;
      const text = document.createElement('p'); text.className = 'precision-card-text'; text.textContent = lane.receipt.outputText;
      const facts = document.createElement('p'); facts.className = 'demo-note';
      const execution = lane.receipt.generationEvidence.backendIdentity;
      facts.textContent = `Math used: ${precisionLabel(execution.activationDtype)} · ${lane.receipt.generatedTokenIds.length} tokens · ${lane.receipt.wallTimingMs.toFixed(0)} ms, slowed by recording`;
      article.append(title, text, facts); results.append(article);
    }
    const save = /** @type {HTMLButtonElement | null} */ ($('live-precision-export'));
    if (save) save.disabled = lanes.length === 0;
    state.generating = false; state.abortController = null; setGenerating(false);
    setPhase(signal.aborted ? 'Comparison stopped' : 'Comparison finished');
  }
}
