// The benchmark page: shows the protocol, this device and the switches, runs
// the benchmark on request, and renders the evidence it returns.

import { loadFlags } from './flags.js';
import { validateProtocol } from './protocol.js';
import { describeDevice, runDeviceBenchmark } from './run.js';
import { give } from './handoff.js';
import { mountComparison } from './compare-results.js';
import { mountNav, pageHref } from './site.js';
import { describeRuntime, loadRuntimeChoice, runtimeSwitch } from './runtimes.js';

const $ = (id) => document.getElementById(id);

const STAGES = [
  ['device', 'Check this device'],
  ['runtime', 'Load the Doppler runtime'],
  ['load', 'Load the model'],
  ['cold', 'First run after loading'],
  ['warm', 'Repeat runs'],
  ['plain', 'Same runs without inspection'],
  ['streaming', 'One streamed run'],
];

const FLAG_LABELS = {
  'benchmark.run': 'Run the benchmark',
  'benchmark.plainPathCheck': 'Cross-check against generation without inspection',
  'views.experimental': 'Experimental views',
  'results.upload': 'Publish a result',
  'rankings.community': 'Community rankings',
  voting: 'Voting',
};

function facts(list, rows) {
  list.replaceChildren(...rows.flatMap(([term, value]) => {
    const dt = document.createElement('dt'); dt.textContent = term;
    const dd = document.createElement('dd'); dd.textContent = value;
    return [dt, dd];
  }));
}

const seconds = (ms) => (Number.isFinite(ms) ? `${(ms / 1000).toFixed(2)} s` : 'not measured');
const millis = (ms) => (Number.isFinite(ms) ? `${ms.toFixed(0)} ms` : 'not measured');
const perSecond = (value) => (Number.isFinite(value) ? `${value.toFixed(1)} tokens/s` : 'not measured');
const gib = (bytes) => (Number.isFinite(bytes) ? `${(bytes / 2 ** 30).toFixed(1)} GiB` : 'not reported');

function showProtocol(protocol) {
  const w = protocol.workload;
  facts($('protocol-facts'), [
    ['Protocol', `${protocol.id} ${protocol.version}`],
    ['Model', w.modelId],
    ['Prompt', w.prompt],
    ['Sampling', `always the most likely token, with no repetition or presence penalty, up to ${w.generation.maxTokens} tokens`],
    ['Runs', `1 after loading, then ${w.warmRuns} repeats`],
    ['Observation', `${w.observationPolicy} (records token IDs and wall time only)`],
    ['Expected answer', protocol.reference.generatedTokenIds ? `${protocol.reference.generatedTokenIds.length} pinned token IDs` : 'not pinned yet'],
  ]);
}

function showDevice(device) {
  const webgpu = { available: 'Available', 'no-adapter': 'Present, but the browser returned no graphics adapter', unavailable: 'Not available in this browser' }[device.webgpu];
  const adapter = device.adapter ? [device.adapter.vendor, device.adapter.architecture, device.adapter.description].filter(Boolean).join(' · ') || 'not described by the browser' : 'none';
  facts($('device-facts'), [
    ['WebGPU', webgpu],
    ['Graphics adapter', adapter],
    ['Largest single GPU buffer', device.limits ? gib(device.limits.maxBufferSize) : 'unknown'],
    // The browser rounds and caps this figure; it is not the GPU's memory.
    ['System memory reported', device.deviceMemoryGb ? `${device.deviceMemoryGb} GB or more (browser-rounded)` : 'not reported by this browser'],
    ['Browser', device.userAgent],
  ]);
}

function showFlags(flags) {
  facts($('flag-facts'), Object.entries(flags).map(([name, flag]) => {
    const state = flag.enabled ? (flag.source === 'preview' ? 'On for this visit (preview)' : 'On') : 'Off';
    const owner = flag.enforcedBy === 'server' ? ' · decided by a server, not by this page' : '';
    return [FLAG_LABELS[name] ?? name, `${state}${owner}`];
  }));
}

function resetStages(includePlain) {
  $('stages').replaceChildren(...STAGES.filter(([id]) => includePlain || !['plain', 'streaming'].includes(id)).map(([id, label]) => {
    const li = document.createElement('li');
    li.dataset.stage = id; li.dataset.state = 'pending'; li.dataset.label = label;
    const mark = document.createElement('span'); mark.textContent = '·';
    const text = document.createElement('span'); text.textContent = label;
    li.append(mark, text);
    return li;
  }));
}

function showStage({ stage, state, detail }) {
  const li = $('stages').querySelector(`[data-stage="${stage}"]`);
  if (!li) return;
  li.dataset.state = state;
  li.firstChild.textContent = { running: '…', done: '✓', failed: '✕' }[state] ?? '·';
  const text = li.lastChild;
  text.textContent = li.dataset.label;
  if (detail) {
    const span = document.createElement('span');
    span.className = 'site-detail';
    span.textContent = ` — ${detail}`;
    text.append(span);
  }
}

function list(element, items) {
  element.replaceChildren(...items.map((item) => { const li = document.createElement('li'); li.textContent = item; return li; }));
  element.hidden = items.length === 0;
}

function showResult(evidence) {
  const { verdict } = evidence;
  $('result').hidden = false;
  const title = { passed: 'Passed', 'not-eligible': 'Completed, but not a comparable result', failed: 'Did not complete' }[verdict.state];
  $('verdict').textContent = title;
  $('verdict').dataset.state = verdict.state;
  // RD's status row: the border style says passed, not comparable or failed.
  $('verdict-row').className = `status-row ${{ passed: 'ready', 'not-eligible': 'warn', failed: 'error' }[verdict.state] ?? ''}`;
  $('verdict-scope').textContent = verdict.state === 'passed'
    ? 'This is evidence about this device, browser and workload. It is a prototype protocol: the number is not a published score and is not ranked against other devices.'
    : '';
  list($('reasons'), verdict.reasons);
  list($('notes'), verdict.notes);

  const summary = verdict.summary;
  const table = $('numbers');
  table.hidden = !summary;
  $('deviation').textContent = '';
  $('answer').hidden = true;
  if (!summary) return;
  const rows = [
    ['First run after loading', seconds(summary.coldWallMs), millis(summary.coldTimeToFirstTokenMs), perSecond(summary.coldDecodeTokensPerSecond)],
    [`Repeat runs (median of ${evidence.warm.length})`, seconds(summary.warmWallMs), millis(summary.warmTimeToFirstTokenMs), perSecond(summary.warmDecodeTokensPerSecond)],
  ];
  if (evidence.plainPath) rows.push([`Without inspection (median of ${evidence.plainPath.length})`, seconds(summary.plainWallMs), '', perSecond(summary.plainDecodeTokensPerSecond)]);
  if (evidence.streaming) rows.push(['Streamed, one run', seconds(summary.streamingWallMs), millis(evidence.streaming.firstChunkMs), '']);
  table.tBodies[0].replaceChildren(...rows.map((cells) => {
    const tr = document.createElement('tr');
    tr.append(...cells.map((cell, column) => { const td = document.createElement('td'); td.textContent = cell; if (column > 0) td.className = 'site-num'; return td; }));
    return tr;
  }));
  const lines = [];
  if (Number.isFinite(summary.warmFastestMs)) lines.push(`Repeat runs ranged from ${seconds(summary.warmFastestMs)} to ${seconds(summary.warmSlowestMs)}.`);
  if (Number.isFinite(summary.plainPathDeviation)) {
    const d = summary.plainPathDeviation;
    lines.push(`Their median was ${(Math.abs(d) * 100).toFixed(1)}% ${d > 0 ? 'longer' : 'shorter'} than the same runs without inspection.`);
  }
  $('deviation').textContent = lines.join(' ');
  if (evidence.cold?.outputText) {
    $('answer').hidden = false;
    $('answer').textContent = evidence.cold.outputText;
    const runs = 1 + evidence.warm.length + (evidence.plainPath?.length ?? 0);
    const checked = summary.referenceMatch === true ? `All ${runs} runs gave the expected ${summary.generatedTokens} tokens.`
      : summary.referenceMatch === false ? 'The answer below is not the expected one.'
      : 'No expected answer is pinned, so the answer below was only checked for being the same on every run.';
    const stopped = evidence.cold.stopReason === 'max-tokens' ? ` The answer stops at the protocol's ${summary.generatedTokens}-token limit, so it ends mid-sentence.` : '';
    lines.push(`${checked}${stopped}`);
    $('deviation').textContent = lines.join(' ');
  }
}

function download(evidence) {
  const blob = new Blob([JSON.stringify(evidence, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `doppler-result-${evidence.run?.id.slice(4, 10) ?? 'failed'}-${evidence.startedAt.slice(0, 19).replace(/[:T]/g, '-')}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function boot() {
  const button = $('run-button');
  const note = $('run-note');
  let protocol; let flags;
  try {
    const response = await fetch(new URL('./config/protocols/device-benchmark.json', import.meta.url), { cache: 'no-store' });
    if (!response.ok) throw new Error(`Protocol file unavailable (HTTP ${response.status}).`);
    protocol = validateProtocol(await response.json());
    flags = await loadFlags();
  } catch (error) {
    note.textContent = `This page could not start: ${error.message}`;
    return;
  }
  mountNav('run', flags);
  // Results opened from files, and this device's own, side by side. Works without a GPU.
  const comparison = mountComparison({ protocol, flags });
  let runtime;
  try { runtime = await loadRuntimeChoice(flags); } catch (error) { note.textContent = `This page could not start: ${error.message}`; return; }
  $('runtime-name').textContent = describeRuntime(runtime);
  const chooser = runtimeSwitch(runtime);
  if (chooser) $('runtime-choice').replaceChildren('Use ', chooser);
  $('runtime-notice').hidden = !runtime.preview;
  $('runtime-notice').textContent = runtime.preview ? `Preview runtime. ${runtime.notice}` : '';
  showProtocol(protocol);
  showFlags(flags);
  const device = await describeDevice();
  showDevice(device);
  resetStages(flags['benchmark.plainPathCheck']?.enabled === true);

  if (!flags['benchmark.run']?.enabled) { note.textContent = 'Running the benchmark is switched off.'; return; }
  if (device.webgpu !== 'available') {
    note.textContent = device.webgpu === 'unavailable'
      ? 'This browser does not expose WebGPU, so the benchmark cannot run here.'
      : 'This browser has WebGPU but returned no graphics adapter, so the benchmark cannot run here.';
    return;
  }
  note.textContent = 'The first run downloads the model (about 420 MB) and saves it in this browser, for this address only.';
  button.disabled = false;

  let latest = null;
  $('download-button').addEventListener('click', () => { if (latest) download(latest); });
  // The result and its answer travel to the other pages in this tab only.
  $('to-compare').addEventListener('click', () => { if (latest) { comparison.add(latest); $('compare').scrollIntoView({ behavior: 'smooth' }); } });
  $('to-explore').addEventListener('click', () => { if (latest?.run) give('run', latest.run); location.href = pageHref('explore', flags); });
  button.addEventListener('click', async () => {
    button.disabled = true;
    $('result').hidden = true;
    note.textContent = 'Running. Keep this tab in front: a background tab is slowed down by the browser.';
    resetStages(flags['benchmark.plainPathCheck']?.enabled === true);
    latest = await runDeviceBenchmark({ protocol, flags, runtime, onStage: showStage });
    globalThis.__DOPPLER_BENCHMARK_EVIDENCE__ = latest;
    showResult(latest);
    $('to-explore').hidden = !latest.run;
    $('identity').textContent = latest.run ? `This answer is run ${latest.run.id.slice(4, 10)}. The same identity names it in Explore and in the downloaded file.` : '';
    note.textContent = '';
    button.textContent = 'Run again';
    button.disabled = false;
  });
}

boot();
