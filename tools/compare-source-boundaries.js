#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';

const POLICY_SCHEMA = 'doppler.source-boundary-comparison-policy/v1';
const RECEIPT_SCHEMA = 'doppler.source-boundary-comparison/v1';

function sha256(value) {
  return `sha256:${crypto.createHash('sha256').update(value).digest('hex')}`;
}

function resolveRepoPath(repoRoot, relativePath, label) {
  if (typeof relativePath !== 'string' || relativePath.trim() === '') {
    throw new Error(`${label} must be a non-empty repository-relative path.`);
  }
  const resolved = path.resolve(repoRoot, relativePath);
  const relative = path.relative(repoRoot, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} must remain inside the repository.`);
  }
  return resolved;
}

export function readNpyF32(bytes, label = 'NPY artifact') {
  if (!Buffer.isBuffer(bytes) || bytes.length < 12 || bytes.subarray(0, 6).toString('latin1') !== '\x93NUMPY') {
    throw new Error(`${label} must be a NumPy .npy artifact.`);
  }
  const major = bytes[6];
  const headerOffset = major === 1 ? 10 : 12;
  const headerLength = major === 1 ? bytes.readUInt16LE(8) : bytes.readUInt32LE(8);
  const header = bytes.subarray(headerOffset, headerOffset + headerLength).toString('latin1');
  if (!header.includes("'descr': '<f4'") || !header.includes("'fortran_order': False")) {
    throw new Error(`${label} must contain little-endian, C-contiguous float32 data.`);
  }
  const dataOffset = headerOffset + headerLength;
  const dataBytes = bytes.subarray(dataOffset);
  if (dataBytes.byteLength % 4 !== 0) {
    throw new Error(`${label} float32 payload is not 4-byte aligned.`);
  }
  return {
    values: new Float32Array(
      dataBytes.buffer,
      dataBytes.byteOffset,
      dataBytes.byteLength / 4,
    ),
    payloadDigest: sha256(dataBytes),
    fileDigest: sha256(bytes),
  };
}

export function compareFloat32Arrays(source, candidate) {
  if (!(source instanceof Float32Array) || !(candidate instanceof Float32Array)) {
    throw new Error('Boundary comparison requires Float32Array inputs.');
  }
  if (source.length !== candidate.length) {
    throw new Error(`Boundary element-count mismatch: source=${source.length}, candidate=${candidate.length}.`);
  }
  let squaredError = 0;
  let sourceSquared = 0;
  let candidateSquared = 0;
  let dot = 0;
  let maximumAbsoluteError = 0;
  let finite = true;
  for (let index = 0; index < source.length; index += 1) {
    const sourceValue = source[index];
    const candidateValue = candidate[index];
    finite = finite && Number.isFinite(sourceValue) && Number.isFinite(candidateValue);
    const delta = candidateValue - sourceValue;
    squaredError += delta * delta;
    sourceSquared += sourceValue * sourceValue;
    candidateSquared += candidateValue * candidateValue;
    dot += sourceValue * candidateValue;
    maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(delta));
  }
  return {
    elementCount: source.length,
    finite,
    exact: squaredError === 0,
    rmse: Math.sqrt(squaredError / Math.max(source.length, 1)),
    relativeRmse: Math.sqrt(squaredError / Math.max(sourceSquared, 1e-30)),
    cosineSimilarity: dot / Math.max(Math.sqrt(sourceSquared * candidateSquared), 1e-30),
    maximumAbsoluteError,
  };
}

function validatePolicy(policy) {
  if (policy?.schema !== POLICY_SCHEMA) {
    throw new Error(`Boundary comparison policy schema must be ${POLICY_SCHEMA}.`);
  }
  if (!Number.isInteger(policy.generationStep) || policy.generationStep < 0) {
    throw new Error('Boundary comparison generationStep must be a non-negative integer.');
  }
  if (!Array.isArray(policy.layers) || policy.layers.some((layer) => !Number.isInteger(layer) || layer < 0)) {
    throw new Error('Boundary comparison layers must contain non-negative integers.');
  }
  if (
    policy.candidateSelection != null
    && policy.candidateSelection !== 'exact'
    && policy.candidateSelection !== 'last-row'
  ) {
    throw new Error('Boundary comparison candidateSelection must be "exact" or "last-row".');
  }
  const tolerance = policy.tolerance;
  if (
    typeof tolerance?.id !== 'string'
    || !Number.isFinite(tolerance.maximumRelativeRmse)
    || !Number.isFinite(tolerance.minimumCosineSimilarity)
    || !Number.isFinite(tolerance.maximumAbsoluteError)
  ) {
    throw new Error('Boundary comparison tolerance contract is incomplete.');
  }
}

function expandMappings(policy) {
  const applyDefaults = (mapping) => ({
    ...mapping,
    candidateSelection: mapping.candidateSelection ?? policy.candidateSelection ?? 'exact',
  });
  const mappings = (policy.globalMappings ?? []).map(applyDefaults);
  for (const layer of policy.layers) {
    for (const mapping of policy.layerMappings ?? []) {
      mappings.push(applyDefaults({
        id: `${mapping.id}-layer-${layer}`,
        layer,
        sourceBoundaryId: mapping.sourceBoundaryTemplate.replaceAll('{L}', String(layer)),
        candidatePath: `layer_${layer}/${mapping.candidateFile}`,
      }));
    }
  }
  return mappings;
}

export function selectCandidateBoundaryValues(source, candidate, selection = 'exact') {
  if (selection === 'exact') {
    return candidate;
  }
  if (selection === 'last-row') {
    if (source.length === 0 || candidate.length < source.length || candidate.length % source.length !== 0) {
      throw new Error(
        `Boundary last-row selection requires candidate length to be a positive multiple of source length: `
        + `source=${source.length}, candidate=${candidate.length}.`
      );
    }
    return candidate.subarray(candidate.length - source.length);
  }
  throw new Error(`Unsupported boundary candidate selection: ${selection}.`);
}

export async function buildBoundaryComparisonReceipt(policyPath, repoRoot = process.cwd()) {
  const resolvedPolicyPath = resolveRepoPath(repoRoot, policyPath, 'policy path');
  const policyBytes = await fs.readFile(resolvedPolicyPath);
  const policy = JSON.parse(policyBytes.toString('utf8'));
  validatePolicy(policy);
  const sourceTranscriptPath = resolveRepoPath(repoRoot, policy.sourceTranscript, 'sourceTranscript');
  const candidateReportPath = resolveRepoPath(repoRoot, policy.candidateReport, 'candidateReport');
  const candidateRoot = resolveRepoPath(repoRoot, policy.candidateRoot, 'candidateRoot');
  const [sourceBytes, candidateReportBytes] = await Promise.all([
    fs.readFile(sourceTranscriptPath),
    fs.readFile(candidateReportPath),
  ]);
  const sourceTranscript = JSON.parse(sourceBytes.toString('utf8'));
  const candidateReport = JSON.parse(candidateReportBytes.toString('utf8'));
  const boundaryById = new Map(
    (sourceTranscript.boundaries ?? []).map((boundary) => [boundary.boundaryId, boundary]),
  );
  const comparisons = [];
  for (const mapping of expandMappings(policy)) {
    const sourceBoundary = boundaryById.get(mapping.sourceBoundaryId);
    if (!sourceBoundary) {
      throw new Error(`Source transcript does not contain boundary ${mapping.sourceBoundaryId}.`);
    }
    if (sourceBoundary.phase !== policy.phase || sourceBoundary.generationStep !== policy.generationStep) {
      throw new Error(`Source boundary ${mapping.sourceBoundaryId} has the wrong phase or generation step.`);
    }
    const sourceArtifactPath = resolveRepoPath(repoRoot, sourceBoundary.artifact?.path, 'source boundary artifact');
    const candidateArtifactPath = resolveRepoPath(candidateRoot, mapping.candidatePath, 'candidate boundary artifact');
    const [sourceArtifactBytes, candidateArtifactBytes] = await Promise.all([
      fs.readFile(sourceArtifactPath),
      fs.readFile(candidateArtifactPath),
    ]);
    const sourceArtifact = readNpyF32(sourceArtifactBytes, mapping.sourceBoundaryId);
    const candidateArtifact = readNpyF32(candidateArtifactBytes, mapping.candidatePath);
    if (sourceArtifact.payloadDigest !== sourceBoundary.fullTensorDigest) {
      throw new Error(`Source boundary artifact digest mismatch for ${mapping.sourceBoundaryId}.`);
    }
    const selectedCandidateValues = selectCandidateBoundaryValues(
      sourceArtifact.values,
      candidateArtifact.values,
      mapping.candidateSelection,
    );
    const metrics = compareFloat32Arrays(sourceArtifact.values, selectedCandidateValues);
    const withinTolerance = metrics.finite
      && metrics.relativeRmse <= policy.tolerance.maximumRelativeRmse
      && metrics.cosineSimilarity >= policy.tolerance.minimumCosineSimilarity
      && metrics.maximumAbsoluteError <= policy.tolerance.maximumAbsoluteError;
    comparisons.push({
      id: mapping.id,
      layer: mapping.layer ?? null,
      sourceBoundaryId: mapping.sourceBoundaryId,
      candidateArtifact: path.relative(repoRoot, candidateArtifactPath).split(path.sep).join('/'),
      candidateSelection: mapping.candidateSelection,
      sourcePayloadDigest: sourceArtifact.payloadDigest,
      candidatePayloadDigest: candidateArtifact.payloadDigest,
      metrics,
      withinTolerance,
    });
  }
  const firstExactDivergence = comparisons.find((comparison) => !comparison.metrics.exact) ?? null;
  const firstToleranceDivergence = comparisons.find((comparison) => !comparison.withinTolerance) ?? null;
  const sourceParity = candidateReport.metrics?.sourceParity ?? null;
  return {
    schema: RECEIPT_SCHEMA,
    policy: {
      path: path.relative(repoRoot, resolvedPolicyPath).split(path.sep).join('/'),
      digest: sha256(policyBytes),
      tolerance: policy.tolerance,
    },
    source: {
      transcript: policy.sourceTranscript,
      digest: sha256(sourceBytes),
      model: sourceTranscript.model,
      revision: sourceTranscript.revision,
      execution: sourceTranscript.execution,
    },
    candidate: {
      report: policy.candidateReport,
      digest: sha256(candidateReportBytes),
      modelId: candidateReport.modelId,
      surface: candidateReport.surface,
      deviceInfo: candidateReport.deviceInfo,
      sourceParity,
    },
    capture: {
      phase: policy.phase,
      generationStep: policy.generationStep,
      layers: policy.layers,
      comparisonCount: comparisons.length,
    },
    comparisons,
    finding: {
      firstExactDivergence: firstExactDivergence?.sourceBoundaryId ?? null,
      firstToleranceDivergence: firstToleranceDivergence?.sourceBoundaryId ?? null,
      classification: firstToleranceDivergence === null
        ? 'precision-lane-drift-within-diagnostic-tolerance'
        : 'semantic-or-numerical-boundary-divergence',
      tokenParityPassed: sourceParity?.status === 'passed',
      promotionEligible: sourceParity?.status === 'passed' && firstToleranceDivergence === null,
    },
    author: policy.author,
  };
}

async function main() {
  const argv = process.argv.slice(2);
  const policyIndex = argv.indexOf('--policy');
  if (policyIndex < 0 || !argv[policyIndex + 1]) {
    throw new Error('Usage: node tools/compare-source-boundaries.js --policy <path> [--check]');
  }
  const policyPath = argv[policyIndex + 1];
  const receipt = await buildBoundaryComparisonReceipt(policyPath);
  const policy = JSON.parse(await fs.readFile(path.resolve(policyPath), 'utf8'));
  const outputPath = resolveRepoPath(process.cwd(), policy.output, 'output');
  const rendered = `${JSON.stringify(receipt, null, 2)}\n`;
  if (argv.includes('--check')) {
    const existing = await fs.readFile(outputPath, 'utf8');
    if (existing !== rendered) {
      throw new Error(`Boundary comparison receipt is stale: ${policy.output}.`);
    }
    console.log(`boundary comparison: current (${policy.output})`);
    return;
  }
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, rendered, 'utf8');
  console.log(outputPath);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  main().catch((error) => {
    console.error(`[boundary-comparison] ${error.message}`);
    process.exitCode = 1;
  });
}
