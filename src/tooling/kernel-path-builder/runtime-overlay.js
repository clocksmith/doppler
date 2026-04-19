import { isPlainObject } from '../../utils/plain-object.js';

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeTimingLabel(value) {
  return normalizeText(value).toLowerCase();
}

function buildTimingTokens(step) {
  const kernel = normalizeText(step?.kernel).toLowerCase();
  const entry = normalizeText(step?.entry).toLowerCase();
  const op = normalizeText(step?.op).toLowerCase();
  const phase = normalizeText(step?.phase).toLowerCase();
  const section = normalizeText(step?.section).toLowerCase();
  const kernelBase = kernel.replace(/\.wgsl$/u, '');
  return [
    normalizeText(step?.id).toLowerCase(),
    op,
    kernel,
    kernelBase,
    entry,
    `${kernelBase}#${entry}`,
    `${section}.${phase}.${op}`,
    `${section}.${op}`,
  ].filter(Boolean);
}

function scoreTimingLabelForStep(label, step) {
  const normalizedLabel = normalizeTimingLabel(label);
  if (!normalizedLabel) {
    return 0;
  }
  let score = 0;
  for (const token of buildTimingTokens(step)) {
    if (!token) continue;
    if (normalizedLabel === token) {
      score = Math.max(score, 100);
      continue;
    }
    if (normalizedLabel.includes(token)) {
      score = Math.max(score, token.length + 10);
    }
  }
  return score;
}

function buildAllExecutionSteps(model) {
  const steps = [];
  for (const [section, sectionSteps] of Object.entries(model?.execution?.sections ?? {})) {
    for (const step of Array.isArray(sectionSteps) ? sectionSteps : []) {
      steps.push({
        ...step,
        section,
        phase: normalizeText(step?.phase),
      });
    }
  }
  return steps;
}

export function aggregateTopDecodeTimers(decodeProfileSteps, limit = 6) {
  const totals = new Map();
  const steps = Array.isArray(decodeProfileSteps) ? decodeProfileSteps : [];
  for (const step of steps) {
    const timings = step?.timings && typeof step.timings === 'object' ? step.timings : {};
    for (const [key, value] of Object.entries(timings)) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric <= 0) continue;
      totals.set(key, (totals.get(key) ?? 0) + numeric);
    }
  }
  return [...totals.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, limit)
    .map(([label, totalMs]) => ({ label, totalMs }));
}

export function buildKernelPathBuilderRuntimeOverlay(model, report) {
  if (!model || !isPlainObject(report)) {
    return null;
  }
  const metrics = isPlainObject(report.metrics) ? report.metrics : {};
  const gpu = isPlainObject(metrics.gpu) ? metrics.gpu : {};
  const memory = isPlainObject(report.memory) ? report.memory : {};
  const decodeProfileSteps = Array.isArray(metrics.decodeProfileSteps) ? metrics.decodeProfileSteps : [];
  const allSteps = buildAllExecutionSteps(model);
  const stepTimingsById = {};
  const unmatchedTimingLabels = [];

  for (const decodeProfileStep of decodeProfileSteps) {
    const timings = isPlainObject(decodeProfileStep.timings) ? decodeProfileStep.timings : {};
    for (const [label, value] of Object.entries(timings)) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric <= 0) continue;
      let bestStep = null;
      let bestScore = 0;
      for (const step of allSteps) {
        const score = scoreTimingLabelForStep(label, step);
        if (score > bestScore) {
          bestScore = score;
          bestStep = step;
        }
      }
      if (!bestStep || bestScore <= 0) {
        unmatchedTimingLabels.push({ label, totalMs: numeric });
        continue;
      }
      const current = stepTimingsById[bestStep.id] || {
        stepId: bestStep.id,
        section: bestStep.section,
        phase: bestStep.phase || null,
        op: bestStep.op,
        kernel: bestStep.kernel,
        entry: bestStep.entry || 'main',
        totalMs: 0,
        labels: [],
      };
      current.totalMs += numeric;
      current.labels.push({ label, totalMs: numeric });
      stepTimingsById[bestStep.id] = current;
    }
  }

  return {
    source: 'report',
    modelId: normalizeText(report.modelId) || null,
    matchesSelectedModel: normalizeText(report.modelId) === normalizeText(model.modelId),
    timestamp: normalizeText(report.timestamp) || null,
    runtimeProfile: normalizeText(report.runtimeProfile) || null,
    modelLoadMs: metrics.modelLoadMs ?? null,
    firstTokenMs: metrics.firstTokenMs ?? null,
    prefillTokensPerSec: metrics.prefillTokensPerSecTtft ?? metrics.prefillTokensPerSec ?? null,
    decodeTokensPerSec: metrics.decodeTokensPerSec ?? null,
    gpu,
    memory,
    decodeProfileSteps,
    topDecodeTimers: aggregateTopDecodeTimers(decodeProfileSteps),
    executionPlan: isPlainObject(metrics.executionPlan) ? metrics.executionPlan : null,
    kernelPathId: normalizeText(metrics.kernelPathId) || null,
    kernelPathSource: normalizeText(metrics.kernelPathSource) || null,
    stepTimings: Object.values(stepTimingsById).sort((left, right) => right.totalMs - left.totalMs),
    stepTimingsById,
    unmatchedTimingLabels,
  };
}
