import { state } from '../state.js';
import { $, setText } from '../dom.js';
import {
  ENERGY_DEMOS,
  DEFAULT_ENERGY_DEMO_ID,
  ENERGY_METRIC_LABELS,
  VLIW_DATASETS,
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
    summary.hidden = problem !== 'vliw';
  }
  if (bundle) {
    bundle.hidden = problem !== 'vliw';
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
  const energyVliwDataset = $('energy-vliw-dataset');
  const energyVliwBundleLimit = $('energy-vliw-bundle-limit');
  const energyVliwMode = $('energy-vliw-mode');
  const energyVliwScoreMode = $('energy-vliw-score-mode');
  const energyVliwRestarts = $('energy-vliw-restarts');
  const energyVliwTempStart = $('energy-vliw-temp-start');
  const energyVliwTempDecay = $('energy-vliw-temp-decay');
  const energyVliwMutation = $('energy-vliw-mutation');
  const energyVliwSpecSearch = $('energy-vliw-spec-search');
  const energyVliwSpecRestarts = $('energy-vliw-spec-restarts');
  const energyVliwSpecSteps = $('energy-vliw-spec-steps');
  const energyVliwSpecTempStart = $('energy-vliw-spec-temp-start');
  const energyVliwSpecTempDecay = $('energy-vliw-spec-temp-decay');
  const energyVliwSpecMutation = $('energy-vliw-spec-mutation');
  const energyVliwSpecSeed = $('energy-vliw-spec-seed');
  const energyVliwSpecPenalty = $('energy-vliw-spec-penalty');
  const energyVliwSpecLambda = $('energy-vliw-spec-lambda');
  const energyVliwSpecInnerSteps = $('energy-vliw-spec-inner-steps');

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
  if (energyInitMode && defaults.quintel?.initMode) {
    energyInitMode.value = defaults.quintel.initMode;
  }
  if (energyInitSeed && Number.isFinite(defaults.quintel?.initSeed)) {
    energyInitSeed.value = String(defaults.quintel.initSeed);
  }
  if (energyInitScale && Number.isFinite(defaults.quintel?.initScale)) {
    energyInitScale.value = String(defaults.quintel.initScale);
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
  if (energyVliwDataset) {
    energyVliwDataset.innerHTML = '';
    Object.entries(VLIW_DATASETS).forEach(([id, entry]) => {
      const option = document.createElement('option');
      option.value = id;
      option.textContent = entry.label || id;
      energyVliwDataset.appendChild(option);
    });
    if (defaults.vliw?.dataset && VLIW_DATASETS[defaults.vliw.dataset]) {
      energyVliwDataset.value = defaults.vliw.dataset;
    }
  }
  if (energyVliwBundleLimit && Number.isFinite(defaults.vliw?.bundleLimit)) {
    energyVliwBundleLimit.value = String(defaults.vliw.bundleLimit);
  }
  if (energyVliwMode && defaults.vliw?.mode) {
    energyVliwMode.value = defaults.vliw.mode;
  }
  if (energyVliwScoreMode && defaults.vliw?.scoreMode) {
    energyVliwScoreMode.value = defaults.vliw.scoreMode;
  }
  if (energyVliwRestarts && Number.isFinite(defaults.vliw?.restarts)) {
    energyVliwRestarts.value = String(defaults.vliw.restarts);
  }
  if (energyVliwTempStart && Number.isFinite(defaults.vliw?.temperatureStart)) {
    energyVliwTempStart.value = String(defaults.vliw.temperatureStart);
  }
  if (energyVliwTempDecay && Number.isFinite(defaults.vliw?.temperatureDecay)) {
    energyVliwTempDecay.value = String(defaults.vliw.temperatureDecay);
  }
  if (energyVliwMutation && Number.isFinite(defaults.vliw?.mutationCount)) {
    energyVliwMutation.value = String(defaults.vliw.mutationCount);
  }
  if (energyVliwSpecSearch && typeof defaults.vliw?.specSearch?.enabled === 'boolean') {
    energyVliwSpecSearch.checked = defaults.vliw.specSearch.enabled;
  }
  if (energyVliwSpecRestarts && Number.isFinite(defaults.vliw?.specSearch?.restarts)) {
    energyVliwSpecRestarts.value = String(defaults.vliw.specSearch.restarts);
  }
  if (energyVliwSpecSteps && Number.isFinite(defaults.vliw?.specSearch?.steps)) {
    energyVliwSpecSteps.value = String(defaults.vliw.specSearch.steps);
  }
  if (energyVliwSpecTempStart && Number.isFinite(defaults.vliw?.specSearch?.temperatureStart)) {
    energyVliwSpecTempStart.value = String(defaults.vliw.specSearch.temperatureStart);
  }
  if (energyVliwSpecTempDecay && Number.isFinite(defaults.vliw?.specSearch?.temperatureDecay)) {
    energyVliwSpecTempDecay.value = String(defaults.vliw.specSearch.temperatureDecay);
  }
  if (energyVliwSpecMutation && Number.isFinite(defaults.vliw?.specSearch?.mutationCount)) {
    energyVliwSpecMutation.value = String(defaults.vliw.specSearch.mutationCount);
  }
  if (energyVliwSpecSeed && Number.isFinite(defaults.vliw?.specSearch?.seed)) {
    energyVliwSpecSeed.value = String(defaults.vliw.specSearch.seed);
  }
  if (energyVliwSpecPenalty && Number.isFinite(defaults.vliw?.specSearch?.penaltyGate)) {
    energyVliwSpecPenalty.value = String(defaults.vliw.specSearch.penaltyGate);
  }
  if (energyVliwSpecLambda && Number.isFinite(defaults.vliw?.specSearch?.cycleLambda)) {
    energyVliwSpecLambda.value = String(defaults.vliw.specSearch.cycleLambda);
  }
  if (energyVliwSpecInnerSteps && Number.isFinite(defaults.vliw?.specSearch?.innerSteps)) {
    energyVliwSpecInnerSteps.value = String(defaults.vliw.specSearch.innerSteps);
  }
}
