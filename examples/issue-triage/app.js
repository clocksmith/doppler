import { TRIAGE_POLICY, suggestRoute, recordDecision } from './triage.js';
import { openDopplerEngine } from './engines.js';

const $ = (id) => document.getElementById(id);
const DECISIONS_KEY = 'issue-triage-decisions-v1';

function loadDecisions() {
  try {
    return JSON.parse(localStorage.getItem(DECISIONS_KEY) ?? '[]');
  } catch {
    return [];
  }
}
function saveDecisions(decisions) {
  try {
    localStorage.setItem(DECISIONS_KEY, JSON.stringify(decisions));
  } catch {
    // The in-page log still holds them; export before closing.
  }
}

const state = { engine: null, policy: null, controller: null, suggestion: null, decisions: loadDecisions() };

function renderLog() {
  $('log').textContent = state.decisions.slice(-10).reverse()
    .map((d) => `${d.decidedAt}  ${d.issueId}  suggested ${d.suggestedRoute} → ${d.decidedRoute}${d.overridden ? ' (overridden)' : ''}  by ${d.decidedBy}`)
    .join('\n');
}

function renderSuggestion(suggestion) {
  $('suggestion').hidden = false;
  $('suggested').textContent = suggestion.suggestedRoute;
  $('review').textContent = suggestion.needsReview ? 'low confidence: review carefully' : '';
  $('scores').replaceChildren(...[...suggestion.routes].sort((a, b) => b.score - a.score).map((route) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${route.id}</td><td>${route.score.toFixed(3)}</td>`;
    return tr;
  }));
  $('decide').replaceChildren(...state.policy.routes.map((route) => {
    const button = document.createElement('button');
    button.textContent = route.id === suggestion.suggestedRoute ? `Accept: ${route.id}` : `Label as ${route.id}`;
    button.addEventListener('click', () => decide(route.id));
    return button;
  }));
}

function decide(route) {
  const decidedBy = $('decider').value.trim();
  if (!decidedBy) {
    $('status').textContent = 'Enter your name first; every label needs a named human decision.';
    return;
  }
  state.decisions.push(recordDecision(state.suggestion, {
    issueId: $('title').value.trim().slice(0, 80) || '(untitled)', decidedRoute: route, decidedBy,
  }));
  saveDecisions(state.decisions);
  renderLog();
  $('suggestion').hidden = true;
  $('status').textContent = `Recorded ${route}.`;
}

async function suggest() {
  state.controller = new AbortController();
  $('suggest').disabled = true;
  $('cancel').disabled = false;
  $('status').textContent = 'Scoring…';
  const started = performance.now();
  try {
    state.suggestion = await suggestRoute(state.engine.scoreChoices, { title: $('title').value, body: $('body').value },
      { policy: state.policy, signal: state.controller.signal });
    renderSuggestion(state.suggestion);
    $('status').textContent = `Scored in ${Math.round(performance.now() - started)} ms.`;
  } catch (error) {
    // Doppler sessions reject a cancelled request with GenerationError code DOPPLER_GENERATION_ABORTED.
    const cancelled = error.name === 'AbortError' || error.code === 'DOPPLER_GENERATION_ABORTED';
    $('status').textContent = cancelled ? 'Cancelled. The model stays open.' : `Failed: ${error.message}`;
  } finally {
    $('suggest').disabled = false;
    $('cancel').disabled = true;
  }
}

async function main() {
  const descriptor = await (await fetch('./descriptor.json')).json();
  state.policy = { ...TRIAGE_POLICY, reviewMargin: descriptor.reviewMargin ?? null };
  const started = performance.now();
  state.engine = await openDopplerEngine({ capsuleUrl: new URL(descriptor.capsuleUrl, location.href).href,
    openOptions: descriptor.openOptions });
  $('status').textContent = `Model ready in ${Math.round(performance.now() - started)} ms.`;
  $('suggest').disabled = false;
}

$('suggest').addEventListener('click', suggest);
$('cancel').addEventListener('click', () => state.controller?.abort());
$('export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(state.decisions, null, 2)], { type: 'application/json' });
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'triage-decisions.json' });
  a.click();
  URL.revokeObjectURL(a.href);
});
window.addEventListener('pagehide', () => state.engine?.close());
renderLog();
main().catch((error) => { $('status').textContent = `Could not open the model: ${error.message}`; });
