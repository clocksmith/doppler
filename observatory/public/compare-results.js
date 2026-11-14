// Benchmark results from several devices beside each other. Mounted by the
// Benchmark page; every element id is prefixed "cmp-".

import { compareResults, readResult } from './results.js';
import { give } from './handoff.js';
import { bar, cardTable, pageHref } from './site.js';
import { h } from './viz/dom.js';

const $ = (id) => document.getElementById(id);
const state = { protocol: null, flags: null, results: [] };

const rate = (value) => (Number.isFinite(value) ? `${value.toFixed(1)} tokens/s` : 'not measured');
const seconds = (value) => (Number.isFinite(value) ? `${(value / 1000).toFixed(2)} s` : 'not measured');
const millis = (value) => (Number.isFinite(value) ? `${value.toFixed(0)} ms` : 'not measured');
const VERDICTS = { passed: 'Passed', 'not-eligible': 'Not comparable', failed: 'Did not complete', 'other-protocol': 'Other protocol version' };
// RD's status badge: the border style says passed, a warning, or failed.
const VERDICT_STYLE = { passed: 'ready', 'not-eligible': 'warn', failed: 'error', 'other-protocol': 'warn' };

function add(value) {
  const result = readResult(value);
  // The same result opened twice is one result.
  const key = `${result.startedAt}|${result.run?.id ?? ''}`;
  if (!state.results.some((existing) => `${existing.startedAt}|${existing.run?.id ?? ''}` === key)) state.results.push(result);
}

async function openFiles(files) {
  const problems = [];
  let opened = 0;
  for (const file of files) {
    try { add(JSON.parse(await file.text())); opened++; }
    catch (error) { problems.push(`${file.name}: ${error instanceof SyntaxError ? 'This file is not JSON.' : error.message}`); }
  }
  $('cmp-open-status').textContent = [opened ? `Opened ${opened} file${opened === 1 ? '' : 's'}.` : '', ...problems].filter(Boolean).join(' ');
  $('cmp-open').value = '';
  render();
}

function render() {
  const view = compareResults(state.protocol, state.results);
  const any = view.rows.length > 0;
  $('cmp-empty').hidden = any;
  $('cmp-results').hidden = !any;
  $('cmp-results-scroll').hidden = !any;
  $('cmp-clear').hidden = !any;
  $('cmp-summary').textContent = any ? `${view.rows.length} result${view.rows.length === 1 ? '' : 's'} opened, ${view.eligible} passing ${state.protocol.id} ${state.protocol.version}.` : '';
  const longest = Math.max(1, ...view.rows.map((row) => row.relativeDecodeRate ?? 0));
  $('cmp-results').tBodies[0].replaceChildren(...view.rows.map((row, index) => {
    const s = row.summary;
    const relative = row.relativeDecodeRate;
    return h('tr', { data: { comparable: String(row.comparable), state: row.state } },
      h('td', { className: 'site-wrap', title: row.userAgent ?? '' }, h('strong', {}, row.adapter), h('div', {}, row.browser), h('span', { className: 'demo-note' }, `${row.startedAt.replace('T', ' ').slice(0, 16)} UTC${row.runId ? ` · run ${row.runId.slice(4, 10)}` : ''}`)),
      h('td', {}, row.runtime, row.previewRuntime ? [' ', h('span', { className: 'tag', title: 'An unqualified preview build' }, 'preview')] : null, h('span', { className: 'demo-note' }, row.protocol)),
      h('td', {}, h('span', { className: `status-badge ${VERDICT_STYLE[row.state] ?? ''}`, data: { state: row.state } }, VERDICTS[row.state])),
      h('td', { className: 'site-num' }, s ? rate(s.warmDecodeTokensPerSecond) : ''),
      h('td', { className: 'site-num' }, relative === null ? '' : [index === view.basisIndex ? 'the basis ' : `${relative.toFixed(2)}× `, bar(relative / longest)]),
      h('td', { className: 'site-num' }, s ? millis(s.warmTimeToFirstTokenMs) : ''),
      h('td', { className: 'site-num' }, s ? seconds(s.warmWallMs) : ''),
      h('td', { className: 'site-num' }, s ? seconds(s.coldWallMs) : ''),
      h('td', { className: 'site-num' }, Number.isFinite(row.loadMs) ? [seconds(row.loadMs), row.modelWasSaved === false ? h('span', { className: 'demo-note' }, 'with download') : null] : ''),
      h('td', {}, state.results[index].run ? h('button', { className: 'btn btn-small', type: 'button', on: { click: () => { give('run', state.results[index].run); location.href = pageHref('explore', state.flags); } } }, 'Explore') : null),
    );
  }));
  cardTable($('cmp-results'));
  const excluded = view.rows.filter((row) => !row.comparable);
  $('cmp-reasons-section').hidden = excluded.length === 0;
  $('cmp-reasons').replaceChildren(...excluded.map((row) => h('li', {}, h('strong', {}, `${row.adapter}, ${row.browser}: `), row.reasons.join(' '))));
  // Say plainly how little a comparison rests on.
  $('cmp-thin').textContent = !any ? ''
    : view.eligible < 2 ? 'Fewer than two passing results are open, so there is nothing to compare yet.'
      : `The comparison covers ${view.devices} device and browser combination${view.devices === 1 ? '' : 's'} and ${view.runtimes} runtime build${view.runtimes === 1 ? '' : 's'}. `
        + `${view.devices === 1 ? 'Results from one device show how repeatable it is, not how devices differ. ' : ''}`
        + `${view.previews ? `${view.previews} of the passing results ${view.previews === 1 ? 'was' : 'were'} made on a preview runtime that is not qualified. ` : ''}`
        + 'Decoding speed is the median of the repeat runs. Timings are reported by each device; nothing here checks them independently.';
}

export function mountComparison({ protocol, flags }) {
  state.protocol = protocol;
  state.flags = flags;
  $('cmp-open-button').addEventListener('click', () => $('cmp-open').click());
  $('cmp-open').addEventListener('change', (event) => openFiles([...event.target.files]));
  $('cmp-clear').addEventListener('click', () => { state.results = []; $('cmp-open-status').textContent = ''; render(); });
  globalThis.__DOPPLER_COMPARE__ = state;
  render();
  return { add(result) { add(result); render(); } };
}
