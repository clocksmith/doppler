import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { resolveModelTypeCluster } from '../../tools/lib/model-type-taxonomy.js';

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
  assert.deepEqual(
    sourceModel.typeCluster,
    smallestVariant?.typeCluster || null,
    `${sourceModel.sourceCheckpointId}: source type cluster must match its variants`
  );
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
    const catalogEntry = catalogByModelId.get(variant.modelId);
    assert.ok(catalogEntry, `${variant.modelId}: inventory variant must exist in catalog`);
    const expectedTypeCluster = resolveModelTypeCluster(catalogEntry.classification);
    assert.deepEqual(
      variant.typeCluster,
      { id: expectedTypeCluster.id, label: expectedTypeCluster.label },
      `${variant.modelId}: type cluster must be taxonomy-derived`
    );
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
    const isRerankCompare = variant.compare.profile?.kind === 'rerank';
    const expectedCompareCommandCount = variant.compare.profile && variant.compare.benchmarkComparable
      ? (isEmbeddingCompare || isRerankCompare ? 1 : policy.benchmarkCommands.workloads.length * policy.benchmarkCommands.decodeProfiles.length)
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
      } else if (isRerankCompare) {
        assertIncludes(command.command, 'node tools/compare-rerankers.js', `${variant.modelId}: rerank compare command`);
        assertIncludes(command.command, '--load-mode http', `${variant.modelId}: rerank compare command`);
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
      if (variant.nextGate === 'summary-svg' && (isEmbeddingCompare || isRerankCompare)) {
        assert.equal(
          variant.actions.primaryNextCommand,
          null,
          `${variant.modelId}: embedding/rerank summary SVG gates must not point at JSON-only compare commands`
        );
        continue;
      }
      assert.equal(
        variant.actions.primaryNextCommand,
        variant.actions.compareCommands[0]?.command || null,
        `${variant.modelId}: compare gate must use the first policy-generated compare command`
      );
      if (isEmbeddingCompare) {
        assertIncludes(variant.actions.primaryNextCommand, 'node tools/compare-embeddings.js', `${variant.modelId}: compare gate command`);
      } else if (isRerankCompare) {
        assertIncludes(variant.actions.primaryNextCommand, 'node tools/compare-rerankers.js', `${variant.modelId}: compare gate command`);
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

assert.equal(
  Object.values(inventory.summary.typeClusterCounts).reduce((sum, count) => sum + count, 0),
  catalog.models.length,
  'model type counts must cover every catalog lane exactly once'
);

console.log('model-support-inventory-contract.test: ok');
