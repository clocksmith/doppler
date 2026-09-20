import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../', import.meta.url));
export const DEFAULT_ACCEPTANCE_RECORD = 'artifacts/bounded-consolidation-2026-09-20/contracts-checks.json';
const HEX_DIGEST = /^[a-f0-9]{64}$/;
const DIGEST = /^sha256:[a-f0-9]{64}$/;
const PHASES = ['repeat-0', 'repeat-1', 'after-cancellation-and-failed-preparation', 'second-after-first-close'];
const LIFECYCLE_CHECKS = ['cancelled-model-preparation', 'cancellation', 'second-session-survives-first-close'];

function nonempty(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

// Project retained execution evidence, not adoption rows or support labels. This
// validates the summary's evidence shape; it does not rerun or authenticate GPUs.
export function projectPhysicalSummary(summary, { evidencePath, packageSha256 }) {
  const errors = [];
  if (summary?.schema !== 'doppler.startup-verification-physical-summary/v1'
    || !Array.isArray(summary.results) || summary.results.length === 0
    || !HEX_DIGEST.test(summary.package?.sha256 ?? '')
    || !HEX_DIGEST.test(summary.originalReceiptSha256 ?? '')
    || !HEX_DIGEST.test(summary.runnerSha256 ?? '')
    || !nonempty(summary.browserVersion)
    || !Number.isFinite(Date.parse(summary.completedAtUtc))) {
    return { errors: [`${evidencePath}: invalid physical summary identity`], configurations: [] };
  }
  const configurations = summary.results.map((result) => {
    const blockers = [];
    if (summary.package.sha256 !== packageSha256) blockers.push('package-identity-mismatch');
    if (summary.passed !== true || result.passed !== true) blockers.push('execution-not-passed');
    if (!Array.isArray(summary.cleanupErrors) || summary.cleanupErrors.length > 0) blockers.push('cleanup-not-proven');
    if (!nonempty(result.hardware?.vendor) || !nonempty(result.hardware?.architecture)
      || result.hardware?.isFallbackAdapter !== false) blockers.push('physical-device-not-identified');
    if (!nonempty(result.capsule?.capsuleId) || !DIGEST.test(result.capsule?.semanticRoot ?? '')
      || !DIGEST.test(result.capsule?.artifactClosureDigest ?? '')
      || !DIGEST.test(result.targetPlanDigest ?? '')) blockers.push('model-implementation-not-identified');
    if (!['generate', 'embed', 'rerank'].includes(result.operation)) blockers.push('operation-not-supported-by-summary-reader');
    const checks = Array.isArray(result.checks) ? result.checks : [];
    if (LIFECYCLE_CHECKS.some((id) => !checks.some((check) => check.id === id && check.passed === true))
      || checks.some((check) => check.passed !== true)) blockers.push('lifecycle-evidence-incomplete');
    const observations = Array.isArray(result.observations) ? result.observations : [];
    const comparisons = Array.isArray(result.comparisons) ? result.comparisons : [];
    const correctness = PHASES.every((phase) => {
      const observation = observations.find((entry) => entry.phase === phase);
      if (!HEX_DIGEST.test(observation?.outputSha256 ?? '')) return false;
      if (result.operation === 'generate') {
        return observation.matchesFrozenReference === true
          && HEX_DIGEST.test(observation.tokenIdsSha256 ?? '') && observation.tokenCount > 0;
      }
      const comparison = comparisons.find((entry) => entry.phase === phase);
      return comparison?.passed === true && Array.isArray(comparison.checks)
        && comparison.checks.length > 0 && comparison.checks.every((check) => check.passed === true);
    });
    if (!correctness || comparisons.some((entry) => entry.passed !== true
      || !Array.isArray(entry.checks) || entry.checks.length === 0
      || entry.checks.some((check) => check.passed !== true))
      || observations.some((entry) => entry.matchesFrozenReference === false)) blockers.push('correctness-evidence-incomplete');
    return {
      evidencePath,
      originalReceiptPath: summary.originalReceiptPath,
      receiptSha256: summary.originalReceiptSha256,
      runnerSha256: summary.runnerSha256,
      packageSha256: summary.package.sha256,
      packageVersion: summary.package.version,
      host: 'browser',
      hostVersion: summary.browserVersion,
      hardware: result.hardware,
      operation: result.operation,
      capsule: result.capsule,
      targetPlanDigest: result.targetPlanDigest,
      completedAtUtc: summary.completedAtUtc,
      lifecycleChecks: checks.map((check) => check.id),
      requiredCorrectnessPhases: [...PHASES],
      physicalExecutionProven: blockers.length === 0,
      blockers,
      measurementLimits: summary.measurementLimits,
    };
  });
  return { errors, configurations };
}

export async function buildPhysicalAcceptanceReport({
  acceptanceRecord = DEFAULT_ACCEPTANCE_RECORD,
  packageSha256 = null,
} = {}) {
  const errors = [];
  const configurations = [];
  let selectedPackageSha256 = packageSha256;
  try {
    const recordPath = path.resolve(REPO_ROOT, acceptanceRecord);
    const record = JSON.parse(await readFile(recordPath, 'utf8'));
    if (record.schema !== 'doppler.consolidation-contracts-checks/v1'
      || !HEX_DIGEST.test(record.packageSha256 ?? '')
      || !Array.isArray(record.physicalSummaries) || record.physicalSummaries.length === 0) {
      throw new Error('invalid consolidation acceptance record');
    }
    selectedPackageSha256 ??= record.packageSha256;
    if (!HEX_DIGEST.test(selectedPackageSha256)) throw new Error('package SHA-256 must be 64 lowercase hex characters');
    for (const entry of record.physicalSummaries) {
      const summaryPath = path.resolve(path.dirname(recordPath), entry);
      const evidencePath = path.relative(REPO_ROOT, summaryPath);
      const summary = JSON.parse(await readFile(summaryPath, 'utf8'));
      if (summary.package?.sha256 !== record.packageSha256) {
        errors.push(`${evidencePath}: summary does not belong to acceptance archive`);
        continue;
      }
      const projection = projectPhysicalSummary(summary, { evidencePath, packageSha256: selectedPackageSha256 });
      errors.push(...projection.errors);
      configurations.push(...projection.configurations);
    }
  } catch (error) {
    errors.push(`${acceptanceRecord}: ${error.message}`);
  }
  return {
    ok: errors.length === 0,
    errors,
    acceptanceRecord,
    selectedPackageSha256,
    packageSelection: packageSha256 === null ? 'retained-acceptance-archive' : 'explicit-archive',
    checkoutQualified: false,
    configurations,
    scope: 'Retained browser execution summaries, not a GPU rerun, automatic freshness claim, support promotion, or qualification of HEAD. Node/Electron/Bun require their own current-archive acceptance. Adapter evidence does not establish learned-adapter quality.',
  };
}

export async function readDeclaredSupport() {
  const policyPath = 'src/config/support-tiers/subsystems.json';
  const registry = JSON.parse(await readFile(path.join(REPO_ROOT, policyPath), 'utf8'));
  return {
    policyPath,
    declarations: registry.subsystems.filter((entry) => entry.userFacing).map((entry) => ({
      id: entry.id, tier: entry.tier, owner: entry.owner, scope: entry.scope,
      notes: entry.notes, docs: entry.docs,
    })),
    scope: 'Declared subsystem maintenance scope, not blanket host/model qualification. Release support requires matching execution evidence, limitations and regression coverage; adoption is not a prerequisite.',
  };
}
