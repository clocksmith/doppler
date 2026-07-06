import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();

const check = spawnSync(
  process.execPath,
  ['tools/sync-model-support-inventory.js', '--check'],
  {
    cwd: repoRoot,
    encoding: 'utf8',
  }
);

assert.equal(
  check.status,
  0,
  `support inventory check failed\nstdout:\n${check.stdout}\nstderr:\n${check.stderr}`
);

const inventory = JSON.parse(
  fs.readFileSync(path.join(repoRoot, 'benchmarks/vendors/model-support-inventory.json'), 'utf8')
);
const policy = JSON.parse(
  fs.readFileSync(path.join(repoRoot, 'benchmarks/vendors/support-rollout-policy.json'), 'utf8')
);
const catalog = JSON.parse(
  fs.readFileSync(path.join(repoRoot, 'models/catalog.json'), 'utf8')
);
const catalogByModelId = new Map(
  (Array.isArray(catalog?.models) ? catalog.models : [])
    .filter((entry) => typeof entry?.modelId === 'string' && entry.modelId.length > 0)
    .map((entry) => [entry.modelId, entry])
);

const expectedUpdated = [catalog.updatedAt, policy.updated]
  .filter((value) => /^\d{4}-\d{2}-\d{2}$/.test(String(value)))
  .sort()
  .at(-1);
assert.equal(inventory.updated, expectedUpdated, 'inventory updated date must be the latest dated source');

function assertIncludes(value, expected, context) {
  assert.equal(
    String(value).includes(expected),
    true,
    `${context} must include ${expected}: ${value}`
  );
}

function assertRepoRelativeExistingFile(repoPath, context) {
  const value = String(repoPath ?? '');
  assert.equal(value.length > 0, true, `${context} must be a non-empty path`);
  assert.equal(path.isAbsolute(value), false, `${context} must be repo-relative`);
  assert.equal(value.includes('\\'), false, `${context} must use forward slashes`);
  assert.equal(value.split('/').includes('..'), false, `${context} must not escape the repo`);
  assert.equal(fs.existsSync(path.join(repoRoot, value)), true, `${context} must exist: ${value}`);
}

assert.equal(policy.ordering, 'artifact-size-ascending');
assert.equal(policy.preferredArchitecture.selection, 'benchmark-evidence-only');
for (const evidenceId of ['runtimeReport', 'compareResult', 'summarySvg']) {
  assert.equal(
    policy.preferredArchitecture.requiredEvidence.includes(evidenceId),
    true,
    `preferred architecture policy must require ${evidenceId}`
  );
}
assert.equal(policy.benchmarkCommands.mode, 'compute');
assert.equal(policy.benchmarkCommands.save, true);
assert.equal(policy.benchmarkCommands.json, true);
for (const workloadId of ['p064-d064-t0-k1', 'p256-d128-t0-k1', 'p512-d128-t0-k1']) {
  assert.equal(
    policy.benchmarkCommands.workloads.includes(workloadId),
    true,
    `benchmark command policy must include workload ${workloadId}`
  );
}
for (const decodeProfile of ['parity', 'throughput']) {
  assert.equal(
    policy.benchmarkCommands.decodeProfiles.includes(decodeProfile),
    true,
    `benchmark command policy must include decode profile ${decodeProfile}`
  );
}

let previousSize = -1;
for (const sourceModel of inventory.sourceModels) {
  if (sourceModel.minSizeBytes === null) continue;
  assert.equal(
    sourceModel.minSizeBytes >= previousSize,
    true,
    `source models must be sorted by artifact size: ${sourceModel.sourceCheckpointId}`
  );
  previousSize = sourceModel.minSizeBytes;
}

let sourceCommandCount = 0;
for (const catalogEntry of catalogByModelId.values()) {
  const evidence = catalogEntry.benchmarkEvidence;
  if (evidence == null) continue;
  assert.equal(evidence.status, 'benchmark-selected', `${catalogEntry.modelId}: benchmarkEvidence.status`);
  assert.equal(
    typeof evidence.localClaimLaneId === 'string' && evidence.localClaimLaneId.length > 0,
    true,
    `${catalogEntry.modelId}: benchmarkEvidence.localClaimLaneId must be set`
  );
  assertRepoRelativeExistingFile(evidence.runtimeReport, `${catalogEntry.modelId}: benchmarkEvidence.runtimeReport`);
  assertRepoRelativeExistingFile(evidence.compareResult, `${catalogEntry.modelId}: benchmarkEvidence.compareResult`);
  assertRepoRelativeExistingFile(evidence.summarySvg, `${catalogEntry.modelId}: benchmarkEvidence.summarySvg`);
}

for (const sourceModel of inventory.sourceModels) {
  const smallestVariant = sourceModel.variants[0] || null;
  assert.equal(
    sourceModel.nextGate,
    smallestVariant?.nextGate || null,
    `${sourceModel.sourceCheckpointId}: source next gate must come from the smallest listed variant`
  );
  assert.equal(
    sourceModel.nextCommand,
    smallestVariant?.actions?.primaryNextCommand || null,
    `${sourceModel.sourceCheckpointId}: source command must come from the smallest listed variant`
  );
  if (sourceModel.nextCommand) sourceCommandCount += 1;
  if (sourceModel.preferredArchitecture.status !== 'benchmark-selected') continue;
  const selected = sourceModel.variants.find(
    (variant) => variant.modelId === sourceModel.preferredArchitecture.modelId
  );
  assert.ok(selected, `${sourceModel.sourceCheckpointId}: selected architecture must point at a listed variant`);
  assert.equal(selected.compare.benchmarkEvidenceOk, true, `${selected.modelId}: selected architecture needs benchmark evidence`);
  assert.equal(selected.evidence.runtimeReportExists, true, `${selected.modelId}: runtime report must exist`);
  assert.equal(selected.evidence.compareResultExists, true, `${selected.modelId}: compare result must exist`);
  assert.equal(selected.evidence.summarySvgExists, true, `${selected.modelId}: summary SVG must exist`);
  const catalogEntry = catalogByModelId.get(selected.modelId);
  assert.ok(catalogEntry, `${selected.modelId}: selected architecture must exist in models/catalog.json`);
  assert.deepEqual(
    catalogEntry.benchmarkEvidence ?? null,
    {
      status: 'benchmark-selected',
      localClaimLaneId: selected.evidence.localClaimLaneId,
      runtimeReport: selected.evidence.runtimeReport,
      compareResult: selected.evidence.compareResult,
      summarySvg: selected.evidence.summarySvg,
    },
    `${selected.modelId}: models/catalog.json must cite the selected benchmark receipts`
  );
}
assert.equal(sourceCommandCount > 0, true, 'at least one source model must expose a next command recipe');

for (const sourceModel of inventory.sourceModels) {
  for (const variant of sourceModel.variants) {
    assert.ok(variant.actions, `${variant.modelId}: actions must exist`);
    assertIncludes(
      variant.actions.verifyCommand,
      `node tools/run-registry-verify.js ${variant.modelId} --surface auto`,
      `${variant.modelId}: runtime verify command`
    );
    assertIncludes(
      variant.actions.hfDryRunCommand,
      `node tools/publish-hf-registry-model.js --model-id ${variant.modelId} --dry-run`,
      `${variant.modelId}: HF dry-run command`
    );

    const isEmbeddingCompare = variant.compare.profile?.kind === 'embedding';
    const expectedCompareCommandCount = variant.compare.profile && variant.compare.benchmarkComparable
      ? (isEmbeddingCompare ? 1 : policy.benchmarkCommands.workloads.length * policy.benchmarkCommands.decodeProfiles.length)
      : 0;
    assert.equal(
      variant.actions.compareCommands.length,
      expectedCompareCommandCount,
      `${variant.modelId}: compare command count must be policy-derived`
    );
    for (const command of variant.actions.compareCommands) {
      assertIncludes(command.command, `--model-id ${variant.modelId}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, `--warmup ${policy.benchmarkCommands.warmupRuns}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, `--runs ${policy.benchmarkCommands.timedRuns}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, '--save', `${variant.modelId}: compare command`);
      assertIncludes(command.command, '--json', `${variant.modelId}: compare command`);
      if (isEmbeddingCompare) {
        assertIncludes(command.command, 'node tools/compare-embeddings.js', `${variant.modelId}: embedding compare command`);
        assertIncludes(command.command, '--load-mode http', `${variant.modelId}: embedding compare command`);
      } else {
        assertIncludes(command.command, 'node tools/compare-engines.js', `${variant.modelId}: generation compare command`);
        assertIncludes(command.command, `--mode ${policy.benchmarkCommands.mode}`, `${variant.modelId}: compare command`);
      }
    }

    if (variant.nextGate === 'runtime-verify') {
      assert.equal(variant.actions.primaryNextCommand, variant.actions.verifyCommand);
    }
    if (variant.nextGate === 'hf-publish') {
      assert.equal(variant.actions.primaryNextCommand, variant.actions.hfDryRunCommand);
    }
    if (['claim-lane', 'compare-result', 'summary-svg'].includes(variant.nextGate)) {
      assert.equal(
        variant.actions.primaryNextCommand,
        variant.actions.compareCommands[0]?.command || null,
        `${variant.modelId}: compare gate must use the first policy-generated compare command`
      );
      if (isEmbeddingCompare) {
        assertIncludes(variant.actions.primaryNextCommand, 'node tools/compare-embeddings.js', `${variant.modelId}: compare gate command`);
      } else {
        assertIncludes(variant.actions.primaryNextCommand, 'node tools/compare-engines.js', `${variant.modelId}: compare gate command`);
        assertIncludes(variant.actions.primaryNextCommand, '--decode-profile parity', `${variant.modelId}: compare gate command`);
      }
    }
    if ([
      'conversion-config',
      'manifest-weights',
      'compare-profile',
      'benchmark-lane-capability-only',
      'preferred-architecture',
    ].includes(variant.nextGate)) {
      assert.equal(
        variant.actions.primaryNextCommand,
        null,
        `${variant.modelId}: ${variant.nextGate} must not inherit a later-gate command`
      );
    }
  }
}

const qwenEmbedding = inventory.sourceModels
  .flatMap((sourceModel) => sourceModel.variants)
  .find((variant) => variant.modelId === 'qwen-3-embedding-0-6b-q4k-ehf16-af32');
assert.ok(qwenEmbedding, 'qwen-3-embedding-0-6b-q4k-ehf16-af32 must be present in support inventory');
assert.equal(qwenEmbedding.compare.profile?.kind, 'embedding');
assert.equal(qwenEmbedding.compare.profile?.lane, 'performance_comparable');
assert.equal(qwenEmbedding.compare.profile?.tjsModelId, 'onnx-community/Qwen3-Embedding-0.6B-ONNX');
assert.equal(qwenEmbedding.actions.compareCommands.length, 1);
assertIncludes(
  qwenEmbedding.actions.compareCommands[0].command,
  'node tools/compare-embeddings.js --model-id qwen-3-embedding-0-6b-q4k-ehf16-af32',
  'qwen embedding compare command'
);
assert.equal(qwenEmbedding.missing.includes('compare-profile'), false);
assert.equal(qwenEmbedding.nextGate, 'compare-result');

const conversionOnlyIds = new Set(inventory.conversionOnly.map((entry) => entry.modelBaseId));
for (const expected of ['gpt-oss-20b-f16-xmxfp4', 'janus-pro-1b-text-q4k-ehaf16']) {
  assert.equal(conversionOnlyIds.has(expected), true, `${expected} must remain visible as conversion-only`);
}

console.log('model-support-inventory-contract.test: ok');
