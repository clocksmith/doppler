#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(
  REPO_ROOT,
  'tools',
  'policies',
  'electron-design-partner-prospects.json'
);
const EXPECTED = Object.freeze([
  { id: 'anythingllm', applicationName: 'AnythingLLM', wave: 'primary' },
  { id: 'joplin', applicationName: 'Joplin', wave: 'primary' },
  { id: 'cherry-studio', applicationName: 'Cherry Studio', wave: 'primary' },
  { id: 'chatbox', applicationName: 'Chatbox', wave: 'backup' },
  { id: 'affine', applicationName: 'AFFiNE', wave: 'later' },
]);
const RELATIONSHIP_STATUSES = new Set([
  'research-candidate',
  'outreach-authorized',
  'contacted',
  'discovery',
  'pilot-authorized',
  'pilot-active',
  'closed',
]);
const WORKLOADS = new Set([
  'generation',
  'document-intelligence',
  'embedding-retrieval',
  'reranking',
  'provider-compatibility',
]);
const REQUIRED_BLOCKERS = Object.freeze([
  'customer-relationship-unestablished',
  'application-revision-not-pinned',
  'acceptance-suite-not-authorized',
  'paid-production-release-unestablished',
  'customer-fleet-receipts-missing',
  'subsequent-upgrade-unestablished',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function requireStringArray(value, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array`);
    return [];
  }
  const normalized = value.map(normalizeText);
  if (normalized.some((entry) => !entry)) errors.push(`${label} entries must be non-empty strings`);
  if (new Set(normalized).size !== normalized.length) errors.push(`${label} entries must be unique`);
  return normalized;
}

function validatePilot(prospect, errors) {
  const label = prospect.id || '<missing-id>';
  if (!isPlainObject(prospect.pilot)) {
    errors.push(`${label}.pilot must be an object`);
    return;
  }
  if (!normalizeText(prospect.pilot.objective)) errors.push(`${label}.pilot.objective is required`);
  requireStringArray(prospect.pilot.requiredCustomerInputs, `${label}.pilot.requiredCustomerInputs`, errors);
  requireStringArray(prospect.pilot.applicationAcceptance, `${label}.pilot.applicationAcceptance`, errors);
  requireStringArray(prospect.pilot.deliverables, `${label}.pilot.deliverables`, errors);
  if (!normalizeText(prospect.pilot.repeatGate)) errors.push(`${label}.pilot.repeatGate is required`);
}

function validateProviderPolicy(prospect, errors) {
  const label = prospect.id || '<missing-id>';
  const policy = prospect.providerPolicy;
  if (!isPlainObject(policy)) {
    errors.push(`${label}.providerPolicy must be an object`);
    return;
  }
  requireStringArray(policy.candidateIncumbents, `${label}.providerPolicy.candidateIncumbents`, errors);
  if (policy.selectionRule !== 'strongest-qualified-application-outcome') {
    errors.push(`${label}.providerPolicy.selectionRule must preserve provider neutrality`);
  }
  if (policy.doeProof !== 'optional-separately-authorized-provider') {
    errors.push(`${label}.providerPolicy.doeProof must remain optional and separately authorized`);
  }
  if (policy.doeRuntime !== 'eligible-only-after-measured-win') {
    errors.push(`${label}.providerPolicy.doeRuntime must require a measured application win`);
  }
  if (policy.customerActivationAuthority !== true) {
    errors.push(`${label}.providerPolicy.customerActivationAuthority must be true`);
  }
  if (policy.selfPromotionAllowed !== false) {
    errors.push(`${label}.providerPolicy.selfPromotionAllowed must be false`);
  }
}

function validateCustodyPolicy(prospect, errors) {
  const label = prospect.id || '<missing-id>';
  const policy = prospect.custodyPolicy;
  if (!isPlainObject(policy)) {
    errors.push(`${label}.custodyPolicy must be an object`);
    return;
  }
  const expected = {
    rawCustomerContent: 'never-cross-product-by-default',
    customerDerivedEvidence: 'explicit-authorization-required',
    sanitizedRuntimeFailure: 'sanitized-only',
    reproducibleBackendDefect: 'reproducible-evidence-only',
  };
  for (const [field, value] of Object.entries(expected)) {
    if (policy[field] !== value) errors.push(`${label}.custodyPolicy.${field} must be ${value}`);
  }
}

export function validateElectronDesignPartnerProspects(policy) {
  const errors = [];
  if (!isPlainObject(policy)) return { ok: false, errors: ['prospect policy must be an object'] };
  if (policy.schemaVersion !== 1) errors.push('schemaVersion must be 1');
  if (policy.source !== 'doppler') errors.push('source must be doppler');
  if (policy.entryProduct !== 'Doppler Production Release') {
    errors.push('entryProduct must be Doppler Production Release');
  }
  if (policy.recurringProduct !== 'Doppler Release Operations') {
    errors.push('recurringProduct must be Doppler Release Operations');
  }
  if (policy.initialIcp !== 'TypeScript/Electron desktop products on Windows and macOS') {
    errors.push('initialIcp must name the TypeScript/Electron Windows and macOS wedge');
  }
  if (policy.requiredProspects !== EXPECTED.length) {
    errors.push(`requiredProspects must be ${EXPECTED.length}`);
  }
  if (!normalizeText(policy.claimBoundary)) errors.push('claimBoundary is required');
  if (!Array.isArray(policy.prospects)) {
    errors.push('prospects must be an array');
    return { ok: false, errors, prospects: [] };
  }
  if (policy.prospects.length !== EXPECTED.length) {
    errors.push(`prospects must contain exactly ${EXPECTED.length} entries`);
  }
  const ids = new Set();
  for (const [index, prospect] of policy.prospects.entries()) {
    const expected = EXPECTED[index];
    if (!isPlainObject(prospect)) {
      errors.push(`prospects[${index}] must be an object`);
      continue;
    }
    const id = normalizeText(prospect.id);
    if (!id) errors.push(`prospects[${index}].id is required`);
    if (ids.has(id)) errors.push(`${id}: duplicate prospect id`);
    ids.add(id);
    if (expected && id !== expected.id) errors.push(`prospects[${index}].id must be ${expected.id}`);
    if (expected && prospect.applicationName !== expected.applicationName) {
      errors.push(`${id}.applicationName must be ${expected.applicationName}`);
    }
    if (prospect.order !== index + 1) errors.push(`${id}.order must be ${index + 1}`);
    if (expected && prospect.wave !== expected.wave) errors.push(`${id}.wave must be ${expected.wave}`);
    if (!RELATIONSHIP_STATUSES.has(prospect.relationshipStatus)) {
      errors.push(`${id}.relationshipStatus is not recognized`);
    }
    if (prospect.claimAllowed !== false) {
      errors.push(`${id}.claimAllowed must remain false in the prospect register`);
    }
    if (!/^https:\/\/github\.com\/[^/]+\/[^/]+$/.test(normalizeText(prospect.upstreamRepository))) {
      errors.push(`${id}.upstreamRepository must be a canonical GitHub repository URL`);
    }
    const workloads = requireStringArray(prospect.workloads, `${id}.workloads`, errors);
    for (const workload of workloads) {
      if (!WORKLOADS.has(workload)) errors.push(`${id}.workloads contains unsupported ${workload}`);
    }
    validatePilot(prospect, errors);
    validateProviderPolicy(prospect, errors);
    validateCustodyPolicy(prospect, errors);
    const blockers = requireStringArray(prospect.blockers, `${id}.blockers`, errors);
    for (const blocker of REQUIRED_BLOCKERS) {
      if (!blockers.includes(blocker)) errors.push(`${id}.blockers must include ${blocker}`);
    }
  }
  const statusCounts = Object.fromEntries(
    [...RELATIONSHIP_STATUSES].map((status) => [
      status,
      policy.prospects.filter((prospect) => prospect.relationshipStatus === status).length,
    ])
  );
  return {
    ok: errors.length === 0,
    errors,
    prospects: policy.prospects,
    statusCounts,
    primaryProspects: policy.prospects.filter((prospect) => prospect.wave === 'primary').length,
    qualifiedCustomers: 0,
    claimBoundary: policy.claimBoundary,
  };
}

export async function buildElectronDesignPartnerProspectsReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  return {
    policyPath: path.relative(REPO_ROOT, policyPath),
    ...validateElectronDesignPartnerProspects(policy),
  };
}

function formatMarkdown(report) {
  const lines = [
    '# Doppler Electron Design Partner Pipeline',
    '',
    `- contract: ${report.ok ? 'valid' : 'invalid'}`,
    `- researched prospects: ${report.prospects.length}`,
    `- primary wave: ${report.primaryProspects}`,
    `- qualified customers represented here: ${report.qualifiedCustomers}`,
    '',
    report.claimBoundary,
    '',
    '## Ordered targets',
    '',
  ];
  for (const prospect of report.prospects) {
    lines.push(
      `${prospect.order}. **${prospect.applicationName}** — ${prospect.wave}; ${prospect.relationshipStatus}`,
      `   Pilot: ${prospect.pilot.objective}`,
      `   Open gates: ${prospect.blockers.join(', ')}`
    );
  }
  if (report.errors.length > 0) {
    lines.push('', '## Errors', '');
    for (const error of report.errors) lines.push(`- ${error}`);
  }
  return lines.join('\n');
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) throw new Error(`Unknown argument: ${unsupported[0]}`);
  const report = await buildElectronDesignPartnerProspectsReport();
  console.log(json ? JSON.stringify(report, null, 2) : formatMarkdown(report));
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
