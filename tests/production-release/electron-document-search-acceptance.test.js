import fs from 'node:fs/promises';

import {
  APPLICATION_GATE_RECEIPT_SCHEMA,
  hashProductionReleaseEvidence,
} from '../../src/config/production-release-evidence.js';
import { assertProductionRelease } from '../../src/config/production-release.js';

const manifestPath = process.env.DOPPLER_PRODUCTION_RELEASE_PATH
  ?? 'tests/fixtures/production-release/electron-document-search-reranker.json';
const release = assertProductionRelease(JSON.parse(await fs.readFile(manifestPath, 'utf8')));
const receipt = {
  schema: APPLICATION_GATE_RECEIPT_SCHEMA,
  receiptId: 'electron-document-search-acceptance',
  releaseId: release.releaseId,
  applicationRevisionDigest: release.application.revisionDigest,
  workload: release.acceptance.workload,
  oracle: release.acceptance.oracle,
  packSemanticRoot: release.candidate.packSemanticRoot,
  targetPlanId: process.env.DOPPLER_TARGET_PLAN_ID ?? 'webgpu-f32-portable',
  resolvedExecutionId: `sha256:${'f'.repeat(64)}`,
  providerId: 'doppler-webgpu',
  deviceTargetId: process.env.DOPPLER_DEVICE_TARGET_ID
    ?? release.supportedDevices.targets[0].id,
  evaluator: {
    id: 'electron-document-search-reference-evaluator',
    revisionDigest: `sha256:${'e'.repeat(64)}`,
  },
  status: 'passed',
  observations: {
    quality: 0.97,
    coldLatencyMs: 3200,
    warmLatencyMs: 1400,
    peakMemoryBytes: 1610612736,
    failureRate: 0,
    startupPassed: true,
    recoveryPassed: true,
  },
  failedSamples: [],
  createdAtUtc: release.createdAtUtc,
  digest: '',
};
receipt.digest = hashProductionReleaseEvidence(receipt);
process.stdout.write(`${JSON.stringify(receipt)}\n`);
