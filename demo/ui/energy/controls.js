import { state } from '../state.js';
import { $, setText } from '../dom.js';
import {
  ENERGY_DEMOS,
  DEFAULT_ENERGY_DEMO_ID,
  ENERGY_METRIC_LABELS,
} from '../constants.js';

export function updateEnergyStatus(message) {
  const status = $('energy-output-status');
  if (!status) return;
  setText(status, message || 'Idle');
}

export function getEnergyDemoById(id) {
  return ENERGY_DEMOS.find((demo) => demo.id === id) || null;
}

export function setEnergyMetricLabels(problem) {
  const labels = ENERGY_METRIC_LABELS[problem] || ENERGY_METRIC_LABELS.quintel;
  setText($('energy-stat-label-symmetry'), labels.symmetry);
  setText($('energy-stat-label-count'), labels.count);
  setText($('energy-stat-label-binarize'), labels.binarize);
}

export function toggleEnergyProblemControls(problem) {
  const targets = document.querySelectorAll('[data-energy-problem]');
  targets.forEach((element) => {
    const target = element.dataset.energyProblem;
    const matches = target === problem;
    element.hidden = !matches;
  });
  const summary = $('energy-kernel-summary')?.parentElement || null;
  const bundle = $('energy-bundle-view')?.parentElement || null;
  if (summary) {
    summary.hidden = true;
  }
  if (bundle) {
    bundle.hidden = true;
  }
}

export function syncEnergyDemoSelection() {
  const select = $('energy-demo-select');
  if (!select) return;
  const selected = select.value || state.energyDemoId || DEFAULT_ENERGY_DEMO_ID;
  const demo = getEnergyDemoById(selected) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  if (!demo) return;
  state.energyDemoId = demo.id;
  if (select.value !== demo.id) {
    select.value = demo.id;
  }
  setText($('energy-demo-description'), demo.description || '');
  setEnergyMetricLabels(demo.problem || 'quintel');
  toggleEnergyProblemControls(demo.problem || 'quintel');
}

export function populateEnergyDemoSelect() {
  const select = $('energy-demo-select');
  if (!select) return;
  select.innerHTML = '';
  ENERGY_DEMOS.forEach((demo) => {
    const option = document.createElement('option');
    option.value = demo.id;
    option.textContent = demo.label;
    select.appendChild(option);
  });
  const initial = state.energyDemoId || DEFAULT_ENERGY_DEMO_ID;
  const demo = getEnergyDemoById(initial) || getEnergyDemoById(DEFAULT_ENERGY_DEMO_ID);
  if (!demo) return;
  state.energyDemoId = demo.id;
  select.value = demo.id;
  setText($('energy-demo-description'), demo.description || '');
  setEnergyMetricLabels(demo.problem || 'quintel');
  toggleEnergyProblemControls(demo.problem || 'quintel');
  applyEnergyDemoDefaults(demo);
}

export function applyEnergyDemoDefaults(demo) {
  if (!demo || !demo.defaults) return;
  const defaults = demo.defaults;
  const energyQuintelSize = $('energy-quintel-size');
  const energyQuintelThreshold = $('energy-quintel-threshold');
  const energyQuintelCountTarget = $('energy-quintel-count-target');
  const energyRuleMirrorX = $('energy-rule-mirror-x');
  const energyRuleMirrorY = $('energy-rule-mirror-y');
  const energyRuleDiagonal = $('energy-rule-diagonal');
  const energyRuleCount = $('energy-rule-count');
  const energyWeightSymmetry = $('energy-weight-symmetry');
  const energyWeightCount = $('energy-weight-count');
  const energyWeightBinarize = $('energy-weight-binarize');
  const energyInitMode = $('energy-init-mode');
  const energyInitSeed = $('energy-init-seed');
  const energyInitScale = $('energy-init-scale');
  const energySteps = $('energy-steps');
  const energyStepSize = $('energy-step-size');
  const energyGradientScale = $('energy-gradient-scale');
  const energyConvergence = $('energy-convergence');

  if (energyQuintelSize && Number.isFinite(defaults.quintel?.size)) {
    energyQuintelSize.value = String(defaults.quintel.size);
  }
  if (energyQuintelThreshold && Number.isFinite(defaults.quintel?.threshold)) {
    energyQuintelThreshold.value = String(defaults.quintel.threshold);
  }
  if (energyQuintelCountTarget && Number.isFinite(defaults.quintel?.countTarget)) {
    energyQuintelCountTarget.value = String(defaults.quintel.countTarget);
  }
  if (energyRuleMirrorX && typeof defaults.quintel?.mirrorX === 'boolean') {
    energyRuleMirrorX.checked = defaults.quintel.mirrorX;
  }
  if (energyRuleMirrorY && typeof defaults.quintel?.mirrorY === 'boolean') {
    energyRuleMirrorY.checked = defaults.quintel.mirrorY;
  }
  if (energyRuleDiagonal && typeof defaults.quintel?.diagonal === 'boolean') {
    energyRuleDiagonal.checked = defaults.quintel.diagonal;
  }
  if (energyRuleCount && typeof defaults.quintel?.countRule === 'boolean') {
    energyRuleCount.checked = defaults.quintel.countRule;
  }
  if (energyWeightSymmetry && Number.isFinite(defaults.quintel?.symmetryWeight)) {
    energyWeightSymmetry.value = String(defaults.quintel.symmetryWeight);
  }
  if (energyWeightCount && Number.isFinite(defaults.quintel?.countWeight)) {
    energyWeightCount.value = String(defaults.quintel.countWeight);
  }
  if (energyWeightBinarize && Number.isFinite(defaults.quintel?.binarizeWeight)) {
    energyWeightBinarize.value = String(defaults.quintel.binarizeWeight);
  }
  const resolvedInitMode = defaults.init?.mode ?? defaults.quintel?.initMode ?? null;
  const resolvedInitSeed = defaults.init?.seed ?? defaults.quintel?.initSeed ?? null;
  const resolvedInitScale = defaults.init?.scale ?? defaults.quintel?.initScale ?? null;
  if (energyInitMode && resolvedInitMode) {
    energyInitMode.value = resolvedInitMode;
  }
  if (energyInitSeed && Number.isFinite(resolvedInitSeed)) {
    energyInitSeed.value = String(resolvedInitSeed);
  }
  if (energyInitScale && Number.isFinite(resolvedInitScale)) {
    energyInitScale.value = String(resolvedInitScale);
  }
  if (energySteps && Number.isFinite(defaults.loop?.steps)) {
    energySteps.value = String(defaults.loop.steps);
  }
  if (energyStepSize && Number.isFinite(defaults.loop?.stepSize)) {
    energyStepSize.value = String(defaults.loop.stepSize);
  }
  if (energyGradientScale && Number.isFinite(defaults.loop?.gradientScale)) {
    energyGradientScale.value = String(defaults.loop.gradientScale);
  }
  if (energyConvergence && Number.isFinite(defaults.loop?.convergence)) {
    energyConvergence.value = String(defaults.loop.convergence);
  }
}
