#!/usr/bin/env node

import process from 'node:process';
import { pathToFileURL } from 'node:url';
import { buildClaimEvidenceContractReport } from './check-claim-evidence-contract.js';
import { buildCommandSurfaceContractReport } from './check-command-surface-contract.js';
import { buildGoalCompletionReport } from './check-goal-completion.js';
import { buildModelArtifactContractReport } from './check-model-artifact-contract.js';
import { buildPolicySchemaRegistryReport } from './check-policy-schema-registry.js';
import { buildProductIntegrationQualificationReport } from './check-product-integration-qualification.js';
import { buildProviderConformanceReport } from './check-provider-conformance.js';
import { buildRuntimeOwnershipDecisionReport } from './check-runtime-ownership-decisions.js';
import { buildRevocationPropagationReport } from './check-revocation-registry.js';
import { buildRuntimePromotionMonitoringReport } from './check-runtime-promotion-monitoring.js';
import { buildSubsystemSupportContractReport } from './check-subsystem-support-contract.js';
import { SIGNED_REVOCATION_PROTOCOL } from '../src/config/revocation-updates.js';

function parseArgs(argv) {
  const args = {
    json: false,
  };
  for (const token of argv) {
    if (token === '--json') {
      args.json = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

function collectErrors(name, report) {
  return (Array.isArray(report.errors) ? report.errors : []).map((error) => `${name}: ${error}`);
}

function buildSummary(reports) {
  const errors = [
    ...collectErrors('goals', reports.goals),
    ...collectErrors('claim evidence', reports.claimEvidence),
    ...collectErrors('command surface', reports.commandSurface),
    ...collectErrors('model artifact', reports.modelArtifact),
    ...collectErrors('policy schemas', reports.policySchemas),
    ...collectErrors('product integrations', reports.productIntegrations),
    ...collectErrors('provider conformance', reports.providerConformance),
    ...collectErrors('runtime ownership', reports.runtimeOwnership),
    ...collectErrors('revocations', reports.revocations),
    ...collectErrors('promotion monitoring', reports.promotionMonitoring),
    ...collectErrors('subsystem support', reports.subsystemSupport),
  ];
  const signedLiveAuthorityQualified = !reports.goals.goals.some((goal) => (
    goal.blockers.includes('signed-live-revocation-authority-missing')
  ));
  return {
    ok: reports.goals.ok
      && reports.claimEvidence.ok
      && reports.commandSurface.ok
      && reports.modelArtifact.ok
      && reports.policySchemas.ok
      && reports.productIntegrations.ok
      && reports.providerConformance.ok
      && reports.runtimeOwnership.ok
      && reports.revocations.ok
      && reports.promotionMonitoring.ok
      && reports.subsystemSupport.ok,
    errors,
    goals: reports.goals.goals,
    contracts: {
      claimEvidence: {
        ok: reports.claimEvidence.ok,
        claims: reports.claimEvidence.claimCount,
      },
      commandSurface: {
        ok: reports.commandSurface.ok,
        commands: reports.commandSurface.commands.length,
      },
      modelArtifact: {
        ok: reports.modelArtifact.ok,
        catalogModels: reports.modelArtifact.catalogModels,
        registryModels: reports.modelArtifact.registryModels,
      },
      policySchemas: {
        ok: reports.policySchemas.ok,
        policies: reports.policySchemas.policies,
      },
      productIntegrations: {
        ok: reports.productIntegrations.ok,
        gateSatisfied: reports.productIntegrations.gateSatisfied,
        qualified: reports.productIntegrations.qualifiedIntegrations,
        required: 3,
        missingWorkloads: reports.productIntegrations.missingWorkloads,
      },
      providerConformance: {
        ok: reports.providerConformance.ok,
        gateSatisfied: reports.providerConformance.gateSatisfied,
        qualified: reports.providerConformance.qualifiedSuites,
        required: 3,
        missingWorkloads: reports.providerConformance.missingWorkloads,
      },
      runtimeOwnership: {
        ok: reports.runtimeOwnership.ok,
        gateSatisfied: reports.runtimeOwnership.gateSatisfied,
        qualified: reports.runtimeOwnership.qualifiedDecisions,
        required: 3,
        missingWorkloads: reports.runtimeOwnership.missingWorkloads,
      },
      revocations: {
        ok: reports.revocations.ok,
        active: reports.revocations.activeRevocations,
        signatureVerification: reports.revocations.signatureVerification,
        bundled: {
          active: reports.revocations.activeRevocations,
          signatureVerification: reports.revocations.signatureVerification,
        },
        signedLive: {
          authorityQualified: signedLiveAuthorityQualified,
          ...SIGNED_REVOCATION_PROTOCOL,
        },
      },
      promotionMonitoring: {
        ok: reports.promotionMonitoring.ok,
        coverageSatisfied: reports.promotionMonitoring.coverageSatisfied,
        promotions: reports.promotionMonitoring.promotions,
        monitoring: reports.promotionMonitoring.monitoring,
        retained: reports.promotionMonitoring.retained,
        revoked: reports.promotionMonitoring.revoked,
      },
      subsystemSupport: {
        ok: reports.subsystemSupport.ok,
        subsystems: reports.subsystemSupport.subsystems,
        primaryClaims: reports.subsystemSupport.primaryClaims,
      },
    },
  };
}

function formatMarkdown(summary) {
  const lines = [
    '# Doppler Product Readiness Report',
    '',
    `- status: ${summary.ok ? 'ok' : 'invalid'}`,
    '',
    '## Goals',
    '',
  ];
  for (const goal of summary.goals) {
    lines.push(`- ${goal.label}: ${goal.completionPercent}% (${goal.claimableRows}/${goal.rows} rows claimable, ${goal.status})`);
  }
  lines.push(
    '',
    '## Contracts',
    '',
    `- claim evidence: ${summary.contracts.claimEvidence.ok ? 'ok' : 'invalid'} (${summary.contracts.claimEvidence.claims} release claims)`,
    `- command surface: ${summary.contracts.commandSurface.ok ? 'ok' : 'invalid'} (${summary.contracts.commandSurface.commands} commands)`,
    `- model artifact registry: ${summary.contracts.modelArtifact.ok ? 'ok' : 'invalid'} (${summary.contracts.modelArtifact.registryModels}/${summary.contracts.modelArtifact.catalogModels} catalog models exposed)`,
    `- policy schemas: ${summary.contracts.policySchemas.ok ? 'ok' : 'invalid'} (${summary.contracts.policySchemas.policies} policies)`,
    `- maintained application integrations: ${summary.contracts.productIntegrations.gateSatisfied ? 'satisfied' : 'incomplete'} (${summary.contracts.productIntegrations.qualified}/${summary.contracts.productIntegrations.required} qualified; missing ${summary.contracts.productIntegrations.missingWorkloads.join(', ') || 'none'})`,
    `- provider conformance: ${summary.contracts.providerConformance.gateSatisfied ? 'satisfied' : 'incomplete'} (${summary.contracts.providerConformance.qualified}/${summary.contracts.providerConformance.required} qualified; missing ${summary.contracts.providerConformance.missingWorkloads.join(', ') || 'none'})`,
    `- runtime ownership decisions: ${summary.contracts.runtimeOwnership.gateSatisfied ? 'satisfied' : 'incomplete'} (${summary.contracts.runtimeOwnership.qualified}/${summary.contracts.runtimeOwnership.required} qualified; missing ${summary.contracts.runtimeOwnership.missingWorkloads.join(', ') || 'none'})`,
    `- revocation propagation: ${summary.contracts.revocations.ok ? 'ok' : 'invalid'} (bundled ${summary.contracts.revocations.bundled.active} active, signature ${summary.contracts.revocations.bundled.signatureVerification}; signed-live mechanism ${summary.contracts.revocations.signedLive.mechanismAvailable ? 'available' : 'missing'}, authority ${summary.contracts.revocations.signedLive.authorityQualified ? 'qualified' : 'incomplete'})`,
    `- post-promotion monitoring: ${summary.contracts.promotionMonitoring.coverageSatisfied ? 'satisfied' : 'incomplete'} (${summary.contracts.promotionMonitoring.promotions} promotions; ${summary.contracts.promotionMonitoring.monitoring} monitoring, ${summary.contracts.promotionMonitoring.retained} retained, ${summary.contracts.promotionMonitoring.revoked} revoked)`,
    `- subsystem support: ${summary.contracts.subsystemSupport.ok ? 'ok' : 'invalid'} (${summary.contracts.subsystemSupport.subsystems} subsystems, ${summary.contracts.subsystemSupport.primaryClaims} primary claims)`,
    ''
  );
  if (summary.errors.length > 0) {
    lines.push('## Errors', '');
    for (const error of summary.errors) {
      lines.push(`- ${error}`);
    }
  }
  return lines.join('\n').trimEnd();
}

export async function buildProductReadinessReport() {
  const reports = {
    goals: await buildGoalCompletionReport(),
    claimEvidence: await buildClaimEvidenceContractReport(),
    commandSurface: await buildCommandSurfaceContractReport(),
    modelArtifact: await buildModelArtifactContractReport(),
    policySchemas: await buildPolicySchemaRegistryReport(),
    productIntegrations: await buildProductIntegrationQualificationReport(),
    providerConformance: await buildProviderConformanceReport(),
    runtimeOwnership: await buildRuntimeOwnershipDecisionReport(),
    revocations: await buildRevocationPropagationReport(),
    promotionMonitoring: await buildRuntimePromotionMonitoringReport(),
    subsystemSupport: await buildSubsystemSupportContractReport(),
  };
  return buildSummary(reports);
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const summary = await buildProductReadinessReport();
  if (args.json) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log(formatMarkdown(summary));
  }
  if (!summary.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
