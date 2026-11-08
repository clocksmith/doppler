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
  const prompt = getPrompt();
  if (!state.model || !state.modelId || !prompt) { status.textContent = 'Load a model and enter a prompt in the composer first.'; return; }
  if (state.generating || state.modelBusy || state.settingsBusy) return;
  if (state.settings.runtimeProfile !== 'profiles/default') { status.textContent = 'Select the Default runtime profile so both paths share the same baseline settings.'; return; }
  const model = /** @type {import('doppler-gpu/tooling/evidence').DopplerModelHandle} */ (state.model);
  const generation = { temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, seed: 0, maxTokens: 16, useChatTemplate: true };
  state.generating = true;
  state.abortController = new AbortController();
  const signal = state.abortController.signal;
  setGenerating(true); setPhase('Comparing precision');
  results.replaceChildren();
  comparison = { schema: 'doppler.live-precision-comparison/v1', prompt, modelId: state.modelId,
    candidateProfile: 'profiles/demo-f16-comparison', generation, completed: false, lanes: [] };
  /** @type {import('doppler-gpu/tooling/evidence').DopplerModelHandle | null} */
  let candidate = null;
  const lanes = [];
  try {
    status.textContent = 'Running the default path…';
    const baseline = await model.inspect.generate(prompt, { policyId: 'demo/guided-quality', generation: { ...generation, signal } });
    lanes.push({ name: 'Default path', receipt: baseline });
    comparison.lanes = lanes;
    if (!baseline.inputTokenIds?.length) throw new Error('The default path did not record its exact prompt tokens.');
    signal.throwIfAborted();
    status.textContent = 'Opening the declared F16-requested path with the same artifact…';
    const { runtime } = await loadRuntimeProfile('profiles/demo-f16-comparison');
    signal.throwIfAborted();
    const opened = await dr.load(state.modelId, { cache: 'opfs', runtimeConfig: runtime, isolatedLoader: true, signal });
    candidate = opened;
    if (opened.manifestHash !== model.manifestHash) throw new Error('The two paths resolved different artifact bytes.');
    signal.throwIfAborted();
    const observed = await opened.inspect.generate(prompt, { policyId: 'demo/guided-quality',
      inputTokenIds: baseline.inputTokenIds, generation: { ...generation, signal } });
    lanes.push({ name: 'F16-requested path', receipt: observed });
    if (JSON.stringify(observed.inputTokenIds) !== JSON.stringify(baseline.inputTokenIds)) throw new Error('The two paths used different prompt tokens.');
    const index = baseline.generatedTokenIds.findIndex((id, i) => observed.generatedTokenIds[i] !== id);
    const divergence = index >= 0 ? index : baseline.generatedTokenIds.length === observed.generatedTokenIds.length ? null : baseline.generatedTokenIds.length;
    comparison.completed = true; comparison.firstDifferentOutputIndex = divergence;
    status.textContent = divergence === null ? 'Both paths generated the same token IDs.' : `First different output token: ${divergence + 1}. Later differences also include changed context.`;
  } catch (error) {
    comparison.failure = String(error);
    status.textContent = signal.aborted ? 'Comparison stopped; completed lane receipts are retained.'
      : `Comparison unavailable: ${String(error)}. The F16 request is not supported by every model or device.`;
  } finally {
    try { await candidate?.unload(); }
    catch (error) { comparison.cleanupFailure = String(error); status.textContent += ` Cleanup failed: ${String(error)}`; }
    for (const lane of lanes) {
      const article = document.createElement('article'); article.className = 'demo-panel';
      const title = document.createElement('h3'); title.textContent = lane.name;
      const text = document.createElement('p'); text.textContent = lane.receipt.outputText;
      const facts = document.createElement('p');
      const execution = lane.receipt.generationEvidence.backendIdentity;
      facts.textContent = `Resolved activations: ${execution.activationDtype ?? 'not reported'} · ${lane.receipt.generatedTokenIds.length} output tokens · ${lane.receipt.wallTimingMs.toFixed(1)} ms diagnostic wall time. Per-operation precision follows the recorded execution identity.`;
      article.append(title, text, facts); results.append(article);
    }
    const save = /** @type {HTMLButtonElement | null} */ ($('live-precision-export'));
    if (save) save.disabled = lanes.length === 0;
    state.generating = false; state.abortController = null; setGenerating(false);
    setPhase(signal.aborted ? 'Comparison stopped' : 'Precision comparison finished');
  }
}
