/** @param {unknown} value @returns {value is number} */
const measured = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0;

// Read only timestamp measurements already in the receipt. Recorder operation
// counts and phase wall clocks are not substitutes for missing GPU durations.
/** @type {import('./gpu-timings.js').readGpuTimings} */
export function readGpuTimings(receipt) {
  /** @type {any} */
  const evidence = receipt;
  const stats = evidence?.generationEvidence?.stats ?? {};
  const totals = new Map();
  const phases = ['prefill', 'decode'].map((phase) => {
    const records = Array.isArray(stats[`${phase}ProfileSteps`]) ? stats[`${phase}ProfileSteps`] : [];
    const steps = [];
    let missingSteps = 0;
    for (const [index, record] of records.entries()) {
      const kernels = Object.entries(record?.timings ?? {})
        .filter(([, value]) => measured(value)).map(([label, ms]) => ({ label, ms }));
      if (!kernels.length) { missingSteps++; continue; }
      const totalMs = kernels.reduce((sum, kernel) => sum + kernel.ms, 0);
      steps.push({ step: record.step ?? index + 1, kernels, totalMs });
      for (const kernel of kernels) {
        const total = totals.get(kernel.label) ?? { label: kernel.label, prefill: null, decode: null };
        total[phase] = (total[phase] ?? 0) + kernel.ms;
        totals.set(kernel.label, total);
      }
    }
    const reported = stats[phase === 'prefill' ? 'gpuTimePrefillMs' : 'gpuTimeDecodeMs'];
    return { phase, steps, missingSteps, totalMs: steps.length ? steps.reduce((sum, step) => sum + step.totalMs, 0) : null,
      reportedMs: measured(reported) ? reported : null };
  });
  return { phases, kernels: [...totals.values()].sort((a, b) => ((b.prefill ?? 0) + (b.decode ?? 0)) - ((a.prefill ?? 0) + (a.decode ?? 0))) };
}

/** @param {number | null} value */
const ms = (value) => value === null ? 'Not captured' : `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })} ms`;
/** @param {string} tag @param {string} [className] @param {string} [text] */
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

/** @type {import('./gpu-timings.js').renderGpuTimings} */
export function renderGpuTimings(container, receipt) {
  const data = readGpuTimings(receipt);
  const section = node('section', 'xray-card demo-panel demo-panel--inset');
  section.append(node('h3', '', 'GPU work by step'), node('p', 'xray-caption',
    'Each row is one step, split into the GPU programs it ran. The total is GPU time, not elapsed time, and segment order is not the order the programs ran in.'));
  const colors = new Map(data.kernels.map((kernel, index) => [kernel.label, `hsl(${index * 137.5 % 360} 55% 44%)`]));
  if (!data.kernels.length) section.append(node('p', 'xray-empty', 'Tick GPU timing before sending to measure each step.'));
  for (const phase of data.phases) {
    const label = phase.phase === 'prefill' ? 'Read the prompt' : 'Write the answer';
    section.append(node('h4', '', label), node('p', 'xray-caption', `Sum of measured programs: ${ms(phase.totalMs)} · GPU total reported: ${ms(phase.reportedMs)}`));
    if (!phase.steps.length) {
      section.append(node('p', 'xray-empty', `${label}: times per GPU program were not captured.`));
      continue;
    }
    if (phase.missingSteps) section.append(node('p', 'xray-diagnostic', `${phase.missingSteps} steps had no usable times and are left out.`));
    const maximum = Math.max(...phase.steps.map((step) => step.totalMs));
    const waterfall = node('div', 'xray-waterfall');
    waterfall.setAttribute('aria-label', `${label}: time per GPU program, step by step`);
    for (const step of phase.steps) {
      const row = node('div', 'xray-waterfall-row');
      const track = node('div', 'xray-waterfall-segments');
      for (const kernel of step.kernels) {
        const segment = node('span', 'xray-waterfall-segment');
        segment.style.width = `${maximum > 0 ? kernel.ms / maximum * 100 : 0}%`;
        segment.style.backgroundColor = colors.get(kernel.label) ?? '';
        segment.title = `${label} · step ${step.step} · ${kernel.label}: ${ms(kernel.ms)}`;
        segment.setAttribute('aria-label', segment.title);
        track.append(segment);
      }
      row.append(node('span', 'xray-waterfall-step', String(step.step)), track, node('span', 'xray-waterfall-total', ms(step.totalMs)));
      waterfall.append(row);
    }
    section.append(waterfall, node('p', 'xray-caption', `All rows: 0–${ms(maximum)}. Point at a segment for its program and time.`));
  }
  if (data.kernels.length) {
    const details = /** @type {HTMLDetailsElement} */ (node('details', 'xray-kernel-details'));
    details.open = true;
    details.append(node('summary', '', 'Time by GPU program'));
    const table = node('table', 'xray-kernel-table'), head = node('thead'), header = node('tr');
    for (const label of ['GPU program', 'Read the prompt', 'Write the answer', 'Total measured']) { const cell = /** @type {HTMLTableCellElement} */ (node('th', '', label)); cell.scope = 'col'; header.append(cell); }
    head.append(header);table.append(head);
    const body = node('tbody');
    for (const kernel of data.kernels) {
      const row = node('tr'), label = /** @type {HTMLTableCellElement} */ (node('th'));label.scope = 'row';
      const swatch = node('span', 'xray-kernel-swatch');swatch.style.backgroundColor = colors.get(kernel.label) ?? '';
      label.append(swatch, document.createTextNode(kernel.label));
      row.append(label, node('td', '', ms(kernel.prefill)), node('td', '', ms(kernel.decode)), node('td', '', ms((kernel.prefill ?? 0) + (kernel.decode ?? 0))));body.append(row);
    }
    table.append(body);details.append(table);section.append(details);
  }
  container.append(section);
}
