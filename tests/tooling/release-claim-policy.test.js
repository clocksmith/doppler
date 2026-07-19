import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

import {
  checkReleaseClaims,
  validateClaimCatalogContract,
  validateReleaseClaimPolicyShape,
  validateSubsystemPublicClaimBoundaries,
} from '../../tools/check-release-claims.js';

function catalogModel(overrides = {}) {
  return {
    modelId: 'unit-text-model',
    modes: ['text'],
    artifact: {
      format: 'rdrr',
    },
    lifecycle: {
      status: {
        tested: 'verified',
      },
      tested: {
        result: 'pass',
        source: 'runtime-verify',
        lastVerifiedAt: '2026-06-24',
        surface: ['browser', 'node'],
      },
    },
    ...overrides,
  };
}

function releaseClaim(overrides = {}) {
  return {
    modelId: 'unit-text-model',
    mode: 'text',
    surface: ['browser', 'node'],
    verificationSource: 'runtime-verify',
    lastVerifiedAt: '2026-06-24',
    artifactFormat: 'rdrr',
    evidence: {
      kind: 'runtime-verify',
      reportPath: 'reports/unit-text-model/pass.json',
    },
    performanceEvidence: {
      kind: 'runtime-report',
      reportPath: 'reports/unit-text-model/pass.json',
      metricPath: 'metrics.decodeTokensPerSec',
      minValue: 0,
      unit: 'tokens/sec',
    },
    ...overrides,
  };
}

function policy(claims = [releaseClaim()]) {
  return {
    schemaVersion: 1,
    updatedAt: '2026-06-24',
    claims,
  };
}

{
  assert.deepEqual(validateReleaseClaimPolicyShape(policy()), []);
  assert.deepEqual(validateReleaseClaimPolicyShape(policy([releaseClaim({
    mode: 'rerank',
    performanceEvidence: {
      kind: 'rerank-runtime-report',
      reportPath: 'reports/unit-rerank-model/pass.json',
      metricPath: 'metrics.rerankMs',
      minValue: 0,
      unit: 'ms',
    },
  })])), []);
  assert.match(
    validateReleaseClaimPolicyShape(policy([releaseClaim({ surface: [] })])).join('\n'),
    /surface must be a non-empty array/
  );
  assert.match(
    validateReleaseClaimPolicyShape(policy([releaseClaim({
      evidence: {
        kind: 'runtime-verify',
        reportPath: '../escape.json',
      },
    })])).join('\n'),
    /repo-relative JSON path/
  );
}

{
  const errors = validateClaimCatalogContract(
    policy(),
    { models: [catalogModel()] },
    { models: [{ modelId: 'unit-text-model' }] }
  );
  assert.deepEqual(errors, []);
  assert.match(
    validateClaimCatalogContract(
      policy([releaseClaim({ surface: ['node'] })]),
      { models: [catalogModel()] },
      { models: [{ modelId: 'unit-text-model' }] }
    ).join('\n'),
    /surfaces must match lifecycle\.tested\.surface/
  );
  assert.match(
    validateClaimCatalogContract(
      policy([]),
      { models: [catalogModel()] },
      { models: [{ modelId: 'unit-text-model' }] }
    ).join('\n'),
    /missing from release-claim policy/
  );
}

{
  assert.deepEqual(validateSubsystemPublicClaimBoundaries({
    subsystems: [{
      id: 'runtime.text',
      tier: 'tier1',
      claimVisibility: 'primary',
      demoDefault: true,
      docs: ['README.md'],
      entrypoints: ['src/index.js'],
    }],
  }), []);
  assert.match(
    validateSubsystemPublicClaimBoundaries({
      subsystems: [{
        id: 'experimental.demo',
        tier: 'experimental',
        claimVisibility: 'secondary',
        demoDefault: true,
        docs: ['README.md'],
        entrypoints: ['src/experimental/demo.js'],
      }],
    }).join('\n'),
    /only tier1 subsystems may be demoDefault/
  );
}

{
  const tmp = path.join(process.cwd(), `.tmp-release-claims-test-${process.pid}`);
  await fs.rm(tmp, { recursive: true, force: true });
  await fs.mkdir(tmp, { recursive: true });
  const reportPath = path.join(tmp, 'report.json');
  const policyPath = path.join(tmp, 'policy.json');
  const catalogPath = path.join(tmp, 'catalog.json');
  const quickstartPath = path.join(tmp, 'quickstart.json');
  const subsystemsPath = path.join(tmp, 'subsystems.json');
  await fs.writeFile(reportPath, JSON.stringify({
    modelId: 'unit-text-model',
    deviceInfo: {
      adapterInfo: {
        vendor: 'unit',
        architecture: 'test',
      },
    },
    metrics: {
      executionContractArtifact: {
        ok: true,
      },
      generatedText: 'unit generated text',
      decodeTokensPerSec: 1,
    },
  }, null, 2));
  await fs.writeFile(policyPath, JSON.stringify(policy([releaseClaim({
    evidence: {
      kind: 'runtime-verify',
      reportPath: path.relative(process.cwd(), reportPath),
    },
    performanceEvidence: {
      kind: 'runtime-report',
      reportPath: path.relative(process.cwd(), reportPath),
      metricPath: 'metrics.decodeTokensPerSec',
      minValue: 0,
      unit: 'tokens/sec',
    },
  })]), null, 2));
  await fs.writeFile(catalogPath, JSON.stringify({ models: [catalogModel()] }, null, 2));
  await fs.writeFile(quickstartPath, JSON.stringify({ models: [{ modelId: 'unit-text-model' }] }, null, 2));
  await fs.writeFile(subsystemsPath, JSON.stringify({
    subsystems: [{
      id: 'runtime.text',
      tier: 'tier1',
      claimVisibility: 'primary',
      demoDefault: true,
      docs: ['README.md'],
      entrypoints: ['src/index.js'],
    }],
  }, null, 2));

  const result = await checkReleaseClaims({
    policyPath,
    catalogPath,
    quickstartRegistryPath: quickstartPath,
    subsystemsPath,
  });
  assert.equal(result.ok, true, result.errors.join('\n'));

  await fs.writeFile(reportPath, JSON.stringify({
    modelId: 'unit-text-model',
    deviceInfo: {
      adapterInfo: {
        vendor: 'unit',
        architecture: 'test',
      },
    },
    metrics: {
      executionContractArtifact: {
        ok: true,
      },
      decodeTokensPerSec: 1,
    },
  }, null, 2));
  const failed = await checkReleaseClaims({
    policyPath,
    catalogPath,
    quickstartRegistryPath: quickstartPath,
    subsystemsPath,
  });
  assert.equal(failed.ok, false);
  assert.match(failed.errors.join('\n'), /generated output evidence/);
  await fs.rm(tmp, { recursive: true, force: true });
}

{
  const tmp = path.join(process.cwd(), `.tmp-rerank-release-claims-test-${process.pid}`);
  await fs.rm(tmp, { recursive: true, force: true });
  await fs.mkdir(tmp, { recursive: true });
  const reportPath = path.join(tmp, 'report.json');
  const policyPath = path.join(tmp, 'policy.json');
  const catalogPath = path.join(tmp, 'catalog.json');
  const quickstartPath = path.join(tmp, 'quickstart.json');
  const subsystemsPath = path.join(tmp, 'subsystems.json');
  const model = catalogModel({
    modelId: 'unit-rerank-model',
    modes: ['rerank'],
    lifecycle: {
      status: {
        tested: 'verified',
      },
      tested: {
        result: 'pass',
        source: 'runtime-verify',
        lastVerifiedAt: '2026-06-24',
        surface: ['node'],
      },
    },
  });
  await fs.writeFile(reportPath, JSON.stringify({
    modelId: 'unit-rerank-model',
    deviceInfo: {
      adapterInfo: {
        vendor: 'unit',
        architecture: 'test',
      },
    },
    metrics: {
      executionContractArtifact: {
        ok: true,
      },
      semanticPassed: true,
      semanticPairAcc: 1,
      topDocumentIndex: 0,
      rerankMs: 2,
    },
  }, null, 2));
  await fs.writeFile(policyPath, JSON.stringify(policy([releaseClaim({
    modelId: 'unit-rerank-model',
    mode: 'rerank',
    surface: ['node'],
    evidence: {
      kind: 'runtime-verify',
      reportPath: path.relative(process.cwd(), reportPath),
    },
    performanceEvidence: {
      kind: 'rerank-runtime-report',
      reportPath: path.relative(process.cwd(), reportPath),
      metricPath: 'metrics.rerankMs',
      minValue: 0,
      unit: 'ms',
    },
  })]), null, 2));
  await fs.writeFile(catalogPath, JSON.stringify({ models: [model] }, null, 2));
  await fs.writeFile(quickstartPath, JSON.stringify({ models: [] }, null, 2));
  await fs.writeFile(subsystemsPath, JSON.stringify({ subsystems: [] }, null, 2));

  const result = await checkReleaseClaims({
    policyPath,
    catalogPath,
    quickstartRegistryPath: quickstartPath,
    subsystemsPath,
  });
  assert.equal(result.ok, true, result.errors.join('\n'));

  await fs.writeFile(reportPath, JSON.stringify({
    modelId: 'unit-rerank-model',
    deviceInfo: {
      adapterInfo: {
        vendor: 'unit',
        architecture: 'test',
      },
    },
    metrics: {
      executionContractArtifact: {
        ok: true,
      },
      rerankMs: 2,
    },
  }, null, 2));
  const failed = await checkReleaseClaims({
    policyPath,
    catalogPath,
    quickstartRegistryPath: quickstartPath,
    subsystemsPath,
  });
  assert.equal(failed.ok, false);
  assert.match(failed.errors.join('\n'), /semantic rerank evidence/);
  await fs.rm(tmp, { recursive: true, force: true });
}

{
  const tmp = path.join(process.cwd(), `.tmp-sequence-release-claims-test-${process.pid}`);
  await fs.rm(tmp, { recursive: true, force: true });
  await fs.mkdir(tmp, { recursive: true });
  const reportPath = path.join(tmp, 'report.json');
  const policyPath = path.join(tmp, 'policy.json');
  const catalogPath = path.join(tmp, 'catalog.json');
  const quickstartPath = path.join(tmp, 'quickstart.json');
  const subsystemsPath = path.join(tmp, 'subsystems.json');
  const modelId = 'unit-protein-encoder';
  const model = catalogModel({
    modelId,
    modes: ['embedding'],
    lifecycle: {
      status: { tested: 'verified' },
      tested: {
        result: 'pass',
        source: 'runtime-verify',
        lastVerifiedAt: '2026-07-19',
        surface: ['node'],
      },
    },
  });
  const claim = releaseClaim({
    modelId,
    mode: 'embedding',
    surface: ['node'],
    lastVerifiedAt: '2026-07-19',
    evidence: {
      kind: 'runtime-verify',
      reportPath: path.relative(process.cwd(), reportPath),
    },
    performanceEvidence: {
      kind: 'runtime-report',
      reportPath: path.relative(process.cwd(), reportPath),
      metricPath: 'result.phase.totalMs',
      minValue: 0,
      unit: 'ms',
    },
  });
  const sequenceReport = {
    schema: 'doppler.sequenceModelQualification.v1',
    passed: true,
    model: {
      modelId,
      sequence: { tokenEmbeddings: true },
    },
    runtime: {
      adapterInfo: { architecture: 'test' },
    },
    result: {
      checks: [
        'model.identity',
        'sequence.contract',
        'tokenizer.ids',
        'pooledEmbedding.finite',
        'pooledEmbedding.parity',
        'tokenEmbeddings.finite',
        'tokenEmbeddings.parity',
      ].map((id) => ({ id, passed: true })),
      phase: { totalMs: 1 },
    },
  };
  await fs.writeFile(reportPath, JSON.stringify(sequenceReport, null, 2));
  await fs.writeFile(policyPath, JSON.stringify(policy([claim]), null, 2));
  await fs.writeFile(catalogPath, JSON.stringify({ models: [model] }, null, 2));
  await fs.writeFile(quickstartPath, JSON.stringify({ models: [] }, null, 2));
  await fs.writeFile(subsystemsPath, JSON.stringify({ subsystems: [] }, null, 2));

  const result = await checkReleaseClaims({
    policyPath,
    catalogPath,
    quickstartRegistryPath: quickstartPath,
    subsystemsPath,
  });
  assert.equal(result.ok, true, result.errors.join('\n'));

  sequenceReport.result.checks.find((check) => check.id === 'pooledEmbedding.parity').passed = false;
  await fs.writeFile(reportPath, JSON.stringify(sequenceReport, null, 2));
  const failed = await checkReleaseClaims({
    policyPath,
    catalogPath,
    quickstartRegistryPath: quickstartPath,
    subsystemsPath,
  });
  assert.equal(failed.ok, false);
  assert.match(failed.errors.join('\n'), /sequence-parity evidence/);
  await fs.rm(tmp, { recursive: true, force: true });
}

console.log('release-claim-policy.test: ok');
