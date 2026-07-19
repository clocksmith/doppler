#!/usr/bin/env node

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { hashWgslSemanticEvidenceValue } from '../src/tooling/wgsl-repair-semantic-gate.js';

const POLICY_PATH = 'tools/policies/wgsl-writer-v3-training-policy.json';
const EVALUATION_PATH =
  'reports/training/wgsl-writer/doppler-wgsl-writer-v3/evaluation/checkpoint-selection/evaluation.json';
const OUTPUT_PATH =
  'reports/training/wgsl-writer/doppler-wgsl-writer-v3/evaluation/selection/selected-lane.json';

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function sha256File(filePath) {
  return createHash('sha256').update(await fs.readFile(filePath)).digest('hex');
}

function compare(left, right) {
  const fields = [
    'semanticPassRate',
    'qualityPasses',
    'responseContractPasses',
    'compilePasses',
  ];
  for (const field of fields) {
    if (left.summary[field] !== right.summary[field]) {
      return right.summary[field] - left.summary[field];
    }
  }
  if (left.summary.meanResponseCharacters !== right.summary.meanResponseCharacters) {
    return left.summary.meanResponseCharacters - right.summary.meanResponseCharacters;
  }
  return left.candidateId.localeCompare(right.candidateId);
}

export async function selectWgslWriterV3Lane() {
  const policy = await readJson(POLICY_PATH);
  const evaluation = await readJson(EVALUATION_PATH);
  const candidates = evaluation.candidates
    .filter((candidate) => candidate.capabilityAuthority === true)
    .sort(compare);
  if (candidates.length === 0) throw new Error('No authoritative V3 candidate was evaluated.');
  const selected = candidates[0];
  const admitted = selected.summary.semanticPassRate
    >= policy.evaluation.minimumSelectionSemanticPassRate;
  const core = {
    schema: 'doppler.wgsl-writer-v3-lane-selection/v1',
    experimentId: policy.experimentId,
    policy: { path: POLICY_PATH, sha256: await sha256File(POLICY_PATH) },
    evaluation: { path: EVALUATION_PATH, sha256: await sha256File(EVALUATION_PATH) },
    ranking: candidates.map((candidate, index) => ({
      rank: index + 1,
      candidateId: candidate.candidateId,
      summary: candidate.summary,
    })),
    selected: admitted ? {
      candidateId: selected.candidateId,
      adapterPath: selected.adapterPath,
      adapterTreeSha256: selected.adapterTreeSha256,
      summary: selected.summary,
    } : null,
    minimumSemanticPassRate: policy.evaluation.minimumSelectionSemanticPassRate,
    decision: admitted ? 'lane_selected' : 'no_lane_met_selection_gate',
    seedConfirmationAuthority: admitted,
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
  selectWgslWriterV3Lane().then((result) => {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (!result.receipt.seedConfirmationAuthority) process.exitCode = 1;
  }).catch((error) => {
    console.error(error instanceof Error ? error.stack : String(error));
    process.exitCode = 1;
  });
}
