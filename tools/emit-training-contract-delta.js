#!/usr/bin/env node

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

function normalizeString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function normalizeDate(value, fallback = null) {
  const normalized = normalizeString(value);
  if (!normalized) return fallback;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`Invalid date value: ${normalized}`);
  }
  return parsed.toISOString().slice(0, 10);
}

function parseArgs(argv) {
  const parsed = {
    out: './reports/training/contract-delta/latest.json',
    periodStart: null,
    periodEnd: null,
    notes: [],
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] || '');
    if (token === '--out') {
      parsed.out = String(argv[i + 1] || '').trim();
      i += 1;
      continue;
    }
    if (token === '--period-start') {
      parsed.periodStart = String(argv[i + 1] || '').trim();
      i += 1;
      continue;
    }
    if (token === '--period-end') {
      parsed.periodEnd = String(argv[i + 1] || '').trim();
      i += 1;
      continue;
    }
    if (token === '--note') {
      const note = normalizeString(argv[i + 1]);
      if (note) parsed.notes.push(note);
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }

  const out = normalizeString(parsed.out);
  if (!out) {
    throw new Error('--out is required');
  }

  const today = new Date().toISOString().slice(0, 10);
  return {
    out: resolve(out),
    periodStart: normalizeDate(parsed.periodStart, today),
    periodEnd: normalizeDate(parsed.periodEnd, today),
    notes: parsed.notes,
  };
}

function buildPayload(args) {
  const actor = (
    normalizeString(process.env.GITHUB_ACTOR)
    || normalizeString(process.env.USER)
    || 'unknown'
  );
  return {
    schemaVersion: 1,
    artifactType: 'training_contract_delta',
    generatedAt: new Date().toISOString(),
    period: {
      start: args.periodStart,
      end: args.periodEnd,
    },
    commandContract: {
      commands: ['convert', 'debug', 'bench', 'verify'],
      verifySuites: ['kernels', 'inference', 'training', 'diffusion', 'energy'],
      deterministicIntentMapping: {
        debug: 'investigate',
        bench: 'calibrate',
        verify: 'verify',
      },
      trainingSchemaVersion: 1,
    },
    releaseGates: {
      workflow: '.github/workflows/training-contract-release-gate.yml',
      script: 'tools/ci-training-contract-gates.js',
      lanes: [
        'command_surface_contract',
        'suite_routing_fail_closed',
        'schema_validation',
        'training_smoke',
        'rollout_governance',
      ],
      deterministicWorkloadRegistry: 'src/experimental/training/workload-packs/registry.json',
      reportIdPublication: 'tools/publish-training-report-ids.js',
    },
    governance: {
      demoClaimsMustReferenceReportId: true,
      publishWeeklyContractDelta: true,
      redTeamReviewRequiredBeforeLaunch: true,
      policyDoc: 'docs/training-handbook.md',
    },
    notes: args.notes,
    operator: {
      actor,
      ciRunId: normalizeString(process.env.GITHUB_RUN_ID),
      ciRunAttempt: normalizeString(process.env.GITHUB_RUN_ATTEMPT),
      sourceRef: normalizeString(process.env.GITHUB_REF_NAME),
      commitSha: normalizeString(process.env.GITHUB_SHA),
    },
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const payload = buildPayload(args);
  await mkdir(dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  console.error(`[training-contract-delta] wrote ${args.out}`);
}

await main();
