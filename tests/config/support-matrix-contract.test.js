import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  buildCurrentInferenceStatusBuckets,
  parseArgs,
  resolveRowStatus,
  validateCatalogMatrixInputs,
} from '../../tools/sync-model-support-matrix.js';
import { assertManifestArtifactIntegrity } from '../helpers/local-model-fixture.js';

{
  assert.deepEqual(validateCatalogMatrixInputs({
    updatedAt: '2026-03-06',
    models: [
      {
        modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
        family: 'gemma3',
        artifact: {
          format: 'rdrr',
        },
        baseUrl: './local/gemma-3-270m-it-q4k-ehf16-af32',
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: '4efe64a914892e98be50842aeb16c3b648cc68a5',
          path: 'models/gemma-3-270m-it-q4k-ehf16-af32',
        },
        lifecycle: {
          availability: {
            curated: true,
            local: true,
            hf: true,
          },
          status: {
            demo: 'curated',
          },
        },
      },
    ],
  }), []);
}

{
  assert.deepEqual(validateCatalogMatrixInputs({
    updatedAt: '',
    models: [
      {
        modelId: 'broken-model',
        artifact: {
          format: 'zip',
        },
        baseUrl: null,
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: '',
          path: '',
        },
        lifecycle: {
          availability: {
            curated: true,
            local: true,
            hf: true,
          },
          status: {
            demo: 'curated',
          },
        },
      },
      {
        modelId: 'broken-model',
        artifact: {},
        baseUrl: null,
        lifecycle: {
          status: {
            demo: 'local',
          },
        },
      },
    ],
  }), [
    'catalog updatedAt must be a non-empty string',
    'broken-model: family is required',
    'broken-model: artifact.format must be "rdrr" or "direct-source"',
    'broken-model: lifecycle.availability.hf=true requires hf.revision',
    'broken-model: lifecycle.availability.hf=true requires hf.path',
    'broken-model: lifecycle.availability.curated=true requires a repo-local baseUrl',
    'broken-model: lifecycle.status.demo=curated requires a repo-local baseUrl',
    'duplicate catalog modelId: broken-model',
    'broken-model: family is required',
    'broken-model: artifact.format is required',
    'broken-model: lifecycle.status.demo=local requires a local baseUrl',
  ]);
}

{
  const repoRoot = process.cwd();
  const catalogPath = path.join(repoRoot, 'models', 'catalog.json');
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  assert.deepEqual(validateCatalogMatrixInputs(catalog), []);

  for (const entry of catalog.models) {
    const baseUrl = typeof entry?.baseUrl === 'string' ? entry.baseUrl.trim() : '';
    if (!baseUrl.startsWith('./local/')) {
      continue;
    }
    const artifactDir = path.join(repoRoot, 'models', baseUrl.slice(2));
    const manifestPath = path.join(artifactDir, 'manifest.json');
    let manifest = null;
    try {
      manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        throw error;
      }
    }
    if (!manifest) {
      assert.equal(entry?.lifecycle?.availability?.local, true);
      continue;
    }
    const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
    assert.ok(shards.length > 0, `${entry.modelId}: local artifact must define at least one shard`);
    await assertManifestArtifactIntegrity(manifestPath);
  }
}

{
  assert.throws(
    () => parseArgs(['--output', '--check']),
    /Missing value for --output/
  );
}

{
  assert.equal(resolveRowStatus({
    conversionCount: 1,
    runtimeStatus: 'active',
    catalogCount: 1,
    lifecycleTested: 'unknown',
  }), 'verification-pending');

  assert.equal(resolveRowStatus({
    conversionCount: 1,
    runtimeStatus: 'active',
    catalogCount: 1,
    lifecycleTested: 'verified',
  }), 'verified');

  assert.equal(resolveRowStatus({
    conversionCount: 1,
    runtimeStatus: 'active',
    catalogCount: 1,
    lifecycleTested: 'failed',
  }), 'verification-failed');
}

{
  const buckets = buildCurrentInferenceStatusBuckets({
    catalogModels: [
      {
        modelId: 'verified-model',
        family: 'gemma3',
        modes: ['run'],
        sortOrder: 1,
        lifecycle: {
          status: {
            runtime: 'active',
            tested: 'verified',
          },
          tested: {
            result: 'pass',
            lastVerifiedAt: '2026-03-06',
            surface: 'auto',
          },
        },
      },
      {
        modelId: 'unknown-model',
        family: 'gemma3',
        modes: ['run'],
        sortOrder: 2,
        lifecycle: {
          status: {
            runtime: 'active',
            tested: 'unknown',
          },
        },
      },
      {
        modelId: 'failing-model',
        family: 'qwen3',
        modes: ['run'],
        sortOrder: 3,
        lifecycle: {
          status: {
            runtime: 'active',
            tested: 'failing',
          },
          tested: {
            result: 'fail',
            lastVerifiedAt: '2026-03-06',
            notes: 'Loads but produces incoherent output.',
          },
        },
      },
    ],
    quickStartModelIds: ['quickstart-only-model', 'verified-model'],
    rows: [
      {
        family: 'mamba',
        catalogCount: 0,
        runtimeStatus: 'blocked',
        status: 'blocked-runtime',
      },
      {
        family: 'functiongemma',
        catalogCount: 0,
        runtimeStatus: 'active',
        status: 'conversion-ready',
      },
    ],
  });

  assert.equal(buckets.verified.length, 1);
  assert.equal(buckets.verified[0].modelId, 'verified-model');
  assert.equal(buckets.loadsButUnverified.length, 1);
  assert.equal(buckets.loadsButUnverified[0].modelId, 'unknown-model');
  assert.equal(buckets.knownFailing.length, 1);
  assert.equal(buckets.knownFailing[0].modelId, 'failing-model');
  assert.equal(buckets.quickstartOnly.length, 1);
  assert.equal(buckets.quickstartOnly[0].modelId, 'quickstart-only-model');
  assert.equal(buckets.everythingElse.length, 2);
}

console.log('support-matrix-contract.test: ok');
