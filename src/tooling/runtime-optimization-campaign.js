import { cloneJsonValue } from '../formats/clone-json.js';
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { isPlainObject } from '../formats/plain-object.js';

const CHANGE_CLASSES = new Set([
  'scheduling-allocation-cache',
  'numerical-kernel',
  'precision-quantization',
  'model-artifact',
  'adapter',
  'provider-integration',
]);

function assertObject(value, label) {
  if (!isPlainObject(value)) {
    throw new Error(`runtime optimization: ${label} must be an object.`);
  }
  return value;
}

function assertExactKeys(value, keys, label) {
  const allowed = new Set(keys);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`runtime optimization: ${label}.${key} is not supported.`);
    }
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`runtime optimization: ${label} must be a non-empty string.`);
  }
  return value.trim();
}

function assertIntegerRange(value, label, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(
      `runtime optimization: ${label} must be an integer in [${minimum}, ${maximum}].`
    );
  }
  return value;
}

function requiredCommandRuns(contract) {
  const primaryRuns = 2 + (2 * contract.measurement.pairCount);
  const neighboringRuns = (contract.neighboringWorkloads ?? []).reduce(
    (count, guard) => count + 2 + (2 * guard.pairCount),
    0
  );
  return primaryRuns + neighboringRuns;
}

function validateMetricGate(metric, label, measurement) {
  assertObject(metric, label);
  assertExactKeys(metric, ['path', 'direction', 'minImprovementPercent'], label);
  if (
    metric.path !== measurement.metricPath
    || metric.direction !== measurement.direction
    || metric.minImprovementPercent !== measurement.minImprovementPercent
  ) {
    throw new Error(`runtime optimization: ${label.slice('contract.'.length)} must match ` +
      (label.endsWith('expectedMetric')
        ? 'the frozen measurement acceptance metric.'
        : 'the measured end-to-end gate.'));
  }
}

function validateConditionList(campaign, field) {
  const values = campaign[field];
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error(`runtime optimization: campaign.${field} must be a non-empty array.`);
  }
  const unique = new Set();
  values.forEach((value, index) => {
    const normalized = assertString(value, `contract.campaign.${field}[${index}]`);
    if (unique.has(normalized)) {
      throw new Error(`runtime optimization: campaign.${field} contains duplicate entries.`);
    }
    unique.add(normalized);
  });
}

export function validateRuntimeOptimizationCampaign(campaign, contract) {
  assertObject(campaign, 'contract.campaign');
  assertExactKeys(campaign, [
    'owner', 'changeClass', 'causalHypothesis', 'expectedMetric', 'controlMetric',
    'endToEndAcceptanceMetric', 'budgets', 'stoppingRule', 'retryConditions',
    'revocationConditions',
  ], 'contract.campaign');
  assertString(campaign.owner, 'contract.campaign.owner');
  if (!CHANGE_CLASSES.has(campaign.changeClass)) {
    throw new Error(`runtime optimization: unsupported campaign.changeClass "${campaign.changeClass}".`);
  }
  assertString(campaign.causalHypothesis, 'contract.campaign.causalHypothesis');
  validateMetricGate(campaign.expectedMetric, 'contract.campaign.expectedMetric', contract.measurement);

  const controlMetric = assertObject(campaign.controlMetric, 'contract.campaign.controlMetric');
  assertExactKeys(controlMetric, ['path', 'expectation'], 'contract.campaign.controlMetric');
  if (controlMetric.expectation !== 'unchanged') {
    throw new Error('runtime optimization: campaign.controlMetric.expectation must be "unchanged".');
  }
  if (!contract.verification.comparisons.some((comparison) => comparison.path === controlMetric.path)) {
    throw new Error(
      'runtime optimization: campaign.controlMetric.path must be enforced by verification.comparisons.'
    );
  }

  validateMetricGate(
    campaign.endToEndAcceptanceMetric,
    'contract.campaign.endToEndAcceptanceMetric',
    contract.measurement
  );
  const budgets = assertObject(campaign.budgets, 'contract.campaign.budgets');
  assertExactKeys(budgets, ['maxCandidates', 'maxCommandRunsPerCandidate'], 'contract.campaign.budgets');
  if (budgets.maxCandidates !== contract.mutationPolicy.maxCandidates) {
    throw new Error(
      'runtime optimization: campaign.budgets.maxCandidates must match mutationPolicy.maxCandidates.'
    );
  }
  assertIntegerRange(
    budgets.maxCommandRunsPerCandidate,
    'contract.campaign.budgets.maxCommandRunsPerCandidate',
    1,
    4096
  );
  const minimumCommandRuns = requiredCommandRuns(contract);
  if (budgets.maxCommandRunsPerCandidate < minimumCommandRuns) {
    throw new Error(
      'runtime optimization: campaign.budgets.maxCommandRunsPerCandidate must cover ' +
      `the frozen plan (${minimumCommandRuns} command runs).`
    );
  }

  const stoppingRule = assertObject(campaign.stoppingRule, 'contract.campaign.stoppingRule');
  assertExactKeys(stoppingRule, ['kind', 'retainNegativeResults'], 'contract.campaign.stoppingRule');
  const expectedStoppingKind = contract.measurement.sequentialDecision
    ? 'bonferroni-fixed-looks'
    : 'fixed-contract';
  if (stoppingRule.kind !== expectedStoppingKind) {
    throw new Error(
      `runtime optimization: campaign.stoppingRule.kind must be "${expectedStoppingKind}".`
    );
  }
  if (stoppingRule.retainNegativeResults !== true) {
    throw new Error('runtime optimization: campaign.stoppingRule.retainNegativeResults must be true.');
  }

  validateConditionList(campaign, 'retryConditions');
  validateConditionList(campaign, 'revocationConditions');
}

export function finalizeRuntimeOptimizationReceipt(receipt) {
  const core = {
    ...receipt,
    promotion: {
      authority: 'human',
      recommended: receipt.decision?.accepted === true,
      runtimeMutationApplied: false,
      requiredStages: ['shadow', 'canary'],
      revocationConditions: cloneJsonValue(receipt.campaign.revocationConditions),
    },
  };
  return { ...core, receiptHash: computeCanonicalSha256(core) };
}
