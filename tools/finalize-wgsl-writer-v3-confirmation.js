#!/usr/bin/env node

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { hashWgslSemanticEvidenceValue } from '../src/tooling/wgsl-repair-semantic-gate.js';

const POLICY_PATH = 'tools/policies/wgsl-writer-v3-training-policy.json';
const SELECTION_PATH =
  'reports/training/wgsl-writer/doppler-wgsl-writer-v3/evaluation/selection/selected-lane.json';
const EVALUATION_PATH =
  'reports/training/wgsl-writer/doppler-wgsl-writer-v3/evaluation/seed-confirmation/evaluation.json';
const OUTPUT_PATH =
  'reports/training/wgsl-writer/doppler-wgsl-writer-v3/evaluation/confirmation/confirmation.json';

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function sha256File(filePath) {
  return createHash('sha256').update(await fs.readFile(filePath)).digest('hex');
}

export async function finalizeWgslWriterV3Confirmation() {
  const [policy, selection, evaluation] = await Promise.all([
    readJson(POLICY_PATH),
    readJson(SELECTION_PATH),
    readJson(EVALUATION_PATH),
  ]);
  if (selection.decision !== 'lane_selected') {
    throw new Error('WGSL writer v3 confirmation requires a selected lane.');
  }
  const candidates = evaluation.candidates.filter((candidate) => (
    candidate.capabilityAuthority === true
  ));
  const expectedSeeds = policy.evaluation.confirmationSeeds;
  if (candidates.length !== expectedSeeds.length) {
    throw new Error('WGSL writer v3 confirmation candidate count mismatch.');
  }
  const rates = candidates.map((candidate) => candidate.summary.semanticPassRate);
  const mean = rates.reduce((sum, value) => sum + value, 0) / rates.length;
  const perSeedPass = rates.every((rate) => (
    rate >= policy.evaluation.minimumConfirmationPerSeedSemanticPassRate
  ));
  const meanPass = mean >= policy.evaluation.minimumConfirmationMeanSemanticPassRate;
  const confirmed = perSeedPass && meanPass;
  const core = {
    schema: 'doppler.wgsl-writer-v3-confirmation/v1',
    experimentId: policy.experimentId,
    policy: { path: POLICY_PATH, sha256: await sha256File(POLICY_PATH) },
    selection: { path: SELECTION_PATH, sha256: await sha256File(SELECTION_PATH) },
    evaluation: { path: EVALUATION_PATH, sha256: await sha256File(EVALUATION_PATH) },
    selectedLaneId: selection.selected.candidateId,
    candidates: candidates.map((candidate) => ({
      candidateId: candidate.candidateId,
      adapterPath: candidate.adapterPath,
      adapterTreeSha256: candidate.adapterTreeSha256,
      semanticPassRate: candidate.summary.semanticPassRate,
    })),
    thresholds: {
      minimumPerSeedSemanticPassRate: policy.evaluation.minimumConfirmationPerSeedSemanticPassRate,
      minimumMeanSemanticPassRate: policy.evaluation.minimumConfirmationMeanSemanticPassRate,
    },
    meanSemanticPassRate: mean,
    decision: confirmed ? 'lane_seed_confirmed' : 'lane_seed_confirmation_failed',
    dopplerParityAuthority: confirmed,
    promotionAuthority: false,
    generalWgslWriterClaim: false,
    claimBoundary: policy.claimBoundary,
  };
  const receipt = { ...core, receiptHash: hashWgslSemanticEvidenceValue(core) };
  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.writeFile(OUTPUT_PATH, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
  return { outputPath: OUTPUT_PATH, receipt };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  finalizeWgslWriterV3Confirmation().then((result) => {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (!result.receipt.dopplerParityAuthority) process.exitCode = 1;
  }).catch((error) => {
    console.error(error instanceof Error ? error.stack : String(error));
    process.exitCode = 1;
  });
}
