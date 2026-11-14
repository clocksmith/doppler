// Token Performance Concertina: where the time went, one fold per decode step.
// A fold opens into the GPU operations timed inside it. GPU time and wall time
// are kept apart, and timing itself adds work, so none of this is a benchmark.

import { answerIndex, sequencePosition, stepTimings } from '../shared-run.js';
import { $, h, ms, percent, visible } from '../viz/dom.js';
import { gap } from '../viz/gap.js';
import { provenanceMark } from '../viz/provenance.js';
import { operationBreakdown } from './comparison-model.js';
import { timeline } from '../viz/timeline.js';

export function renderConcertina({ run, selection, canRun, why, actions }) {
  const timing = stepTimings(run);
  $('timing').hidden = !timing.available;
  $('kernels').hidden = !timing.available;
  $('time-strip').replaceChildren();
  $('step-detail').replaceChildren();
  $('timing-gap').replaceChildren();
  if (!timing.available) {
    $('timing-note').textContent = '';
    // Not timed is a gap; where this browser can run the model it can be recorded.
    $('timing-gap').replaceChildren(gap({ what: timing.reason, action: run.prompt ? { label: 'Record again with GPU timing', unavailable: canRun ? '' : (why || 'A run is in progress.'), run: () => actions.recordAgain(run, { timing: true }) } : null }));
    return;
  }
  const total = timing.steps.reduce((sum, step) => sum + step.totalMs, 0);
  const longest = Math.max(...timing.steps.map((step) => step.totalMs));
  $('timing-note').replaceChildren(provenanceMark('measured', 'GPU timestamps around each operation.'), ' ',
    `GPU time measured for ${timing.steps.length} decode steps: ${ms(total)} in all`
    + `${Number.isFinite(timing.decodeWallMs) ? `, inside ${ms(timing.decodeWallMs)} of wall time for decoding` : ''}. `
    + `Reading the prompt: ${timing.prefillGpuMs === null ? 'not measured on the GPU by this runtime' : ms(timing.prefillGpuMs)}. Timing itself adds work, so these are not benchmark numbers; use Benchmark for a speed.`);
  const open = answerIndex(run, selection.get().position);
  $('time-strip').replaceChildren(timeline(timing.steps.map((step) => ({
    key: step.tokenIndex, ms: step.totalMs, label: visible(step.tokenText), title: `Step ${step.step}, "${visible(step.tokenText)}": ${ms(step.totalMs)} on the GPU`,
  })), { selected: open, label: 'GPU time per decode step', onPick: (index) => selection.select({ position: index === null ? null : sequencePosition(run, index) }) }));
  const step = timing.steps.find((item) => item.tokenIndex === open);
  if (!step && open !== null) $('step-detail').replaceChildren(h('p', { className: 'demo-note' }, 'No GPU decode step was recorded for this selected token. The first answer token is produced during prefill.'));
  if (step) {
    const breakdown = operationBreakdown(step);
    const inside = breakdown.operations;
    const selectedOp = selection.get().op;
    $('step-detail').replaceChildren(
      h('h4', { className: 'type-label' }, `Inside step ${step.step}, "${visible(step.tokenText)}": ${ms(step.totalMs)} on the GPU`),
      h('div', { className: 'site-scroll' }, h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Operation'), h('th', { className: 'site-num' }, 'GPU time'), h('th', { className: 'site-num' }, 'Share of this step'), h('th', {}, 'Duration'))),
        h('tbody', {}, ...inside.map(({ name, time, share }) => h('tr', { attrs: name === selectedOp ? { 'aria-current': 'true' } : {} }, h('td', {}, h('button', { type: 'button', className: 'btn btn-small', ariaPressed: String(name === selectedOp), on: { click: () => selection.select({ op: name === selectedOp ? null : name }) } }, name)), h('td', { className: 'site-num' }, ms(time)), h('td', { className: 'site-num' }, percent(share)), h('td', {}, h('progress', { max: Math.max(step.totalMs, breakdown.sum, 1e-12), value: time, ariaLabel: `${name} GPU time` }))))))),
      h('p', { className: 'demo-note' }, breakdown.exceedsTotal ? 'The operation sum exceeds the recorded step total. These scopes may overlap; their shares are not a partition.' : `${ms(breakdown.remainder)} of the recorded step total is not assigned to a listed operation. This does not measure CPU work or waiting.`));
  }
  $('timing').tBodies[0].replaceChildren(...timing.steps.map((item) => h('tr', { attrs: item.tokenIndex === open ? { 'aria-current': 'true' } : {} },
    h('td', {}, h('button', { type: 'button', className: 'btn btn-small', ariaPressed: String(item.tokenIndex === open), on: { click: () => selection.select({ position: sequencePosition(run, item.tokenIndex) }) } }, String(item.step))),
    h('td', {}, h('code', {}, visible(item.tokenText))),
    h('td', { className: 'site-num' }, ms(item.totalMs)),
    h('td', { className: 'site-num' }, percent(longest > 0 ? item.totalMs / longest : null)),
  )));
  $('kernels').tBodies[0].replaceChildren(...timing.kernels.slice(0, 12).map((kernel) => h('tr', {},
    h('td', {}, h('code', {}, kernel.name)),
    h('td', { className: 'site-num' }, ms(kernel.ms)),
    h('td', { className: 'site-num' }, percent(total > 0 ? kernel.ms / total : null)),
  )));
}

