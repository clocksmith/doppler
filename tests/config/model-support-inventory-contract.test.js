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

    const expectedCompareCommandCount = variant.compare.profile && variant.compare.benchmarkComparable
      ? policy.benchmarkCommands.workloads.length * policy.benchmarkCommands.decodeProfiles.length
      : 0;
    assert.equal(
      variant.actions.compareCommands.length,
      expectedCompareCommandCount,
      `${variant.modelId}: compare command count must be policy-derived`
    );
    for (const command of variant.actions.compareCommands) {
      assertIncludes(command.command, `--model-id ${variant.modelId}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, `--mode ${policy.benchmarkCommands.mode}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, `--warmup ${policy.benchmarkCommands.warmupRuns}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, `--runs ${policy.benchmarkCommands.timedRuns}`, `${variant.modelId}: compare command`);
      assertIncludes(command.command, '--save', `${variant.modelId}: compare command`);
      assertIncludes(command.command, '--json', `${variant.modelId}: compare command`);
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
      assertIncludes(variant.actions.primaryNextCommand, 'node tools/compare-engines.js', `${variant.modelId}: compare gate command`);
      assertIncludes(variant.actions.primaryNextCommand, '--decode-profile parity', `${variant.modelId}: compare gate command`);
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

const conversionOnlyIds = new Set(inventory.conversionOnly.map((entry) => entry.modelBaseId));
for (const expected of ['gpt-oss-20b-f16-xmxfp4', 'janus-pro-1b-text-q4k-ehaf16']) {
  assert.equal(conversionOnlyIds.has(expected), true, `${expected} must remain visible as conversion-only`);
}

console.log('model-support-inventory-contract.test: ok');
