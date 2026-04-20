import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  assertPromotionReady,
  buildArtifactUploadPlan,
  main,
  parseArgs,
  writeBackLocalCatalog,
} from '../../tools/publish-hf-registry-model.js';

function promotionReadyEntry(overrides = {}) {
  const base = {
    modelId: 'translategemma-4b-it-q4k-ehf16-af32',
    sourceCheckpointId: 'google/translategemma-4b-it',
    weightPackId: 'translategemma-4b-it-q4k-ehf16-af32-wp-catalog-v1',
    manifestVariantId: 'translategemma-4b-it-q4k-ehf16-af32-mv-exec-v1',
    artifactCompleteness: 'complete',
    runtimePromotionState: 'manifest-owned',
    weightsRefAllowed: false,
    lifecycle: {
      availability: {
        hf: true,
      },
      status: {
        runtime: 'active',
        tested: 'verified',
      },
      tested: {
        contracts: {
          executionContractOk: true,
        },
      },
    },
  };
  return {
    ...base,
    ...overrides,
    lifecycle: overrides.lifecycle ?? base.lifecycle,
  };
}

{
  const args = parseArgs([
    '--model-id', 'translategemma-4b-it-q4k-ehf16-af32',
    '--catalog-file', 'models/catalog.json',
    '--dry-run',
  ]);
  assert.equal(args.modelId, 'translategemma-4b-it-q4k-ehf16-af32');
  assert.ok(args.catalogFile.endsWith('models/catalog.json'));
  assert.equal(args.dryRun, true);
  assert.equal(args.manifestOnly, false);
  assert.throws(
    () => parseArgs(['--model-id', 'translategemma-4b-it-q4k-ehf16-af32', '--repo-id']),
    /Missing value for --repo-id/
  );
}

{
  const args = parseArgs([
    '--model-id', 'translategemma-4b-it-q4k-ehf16-af32',
    '--manifest-only',
  ]);
  assert.equal(args.manifestOnly, true);
}

{
  const args = parseArgs([
    '--model-id', 'translategemma-4b-it-q4k-ehf16-af32',
    '--bootstrap',
  ]);
  assert.equal(args.bootstrap, true);
  assert.equal(args.manifestOnly, false);
}

{
  const plan = buildArtifactUploadPlan({
    modelId: 'translategemma-4b-it-q4k-ehf16-af32',
    hf: {
      repoId: 'Clocksmith/rdrr',
      path: 'models/translategemma-4b-it-q4k-ehf16-af32',
    },
  }, {
    localDir: '/tmp/translategemma',
  });
  assert.equal(plan.repoId, 'Clocksmith/rdrr');
  assert.equal(plan.targetPath, 'models/translategemma-4b-it-q4k-ehf16-af32');
  assert.equal(plan.manifestDir, '/tmp/translategemma');
}

{
  assert.throws(
    () => buildArtifactUploadPlan({
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      hf: {
        repoId: 'Clocksmith/rdrr',
        path: '',
      },
    }),
    /hf\.path is required to publish/
  );
}

{
  assert.doesNotThrow(() => assertPromotionReady(promotionReadyEntry()));
}

{
  assert.throws(
    () => assertPromotionReady(promotionReadyEntry({
      lifecycle: {
        availability: {
          hf: false,
        },
        status: {
          runtime: 'active',
          tested: 'verified',
        },
        tested: {
          contracts: {
            executionContractOk: true,
          },
        },
      },
    })),
    /lifecycle\.availability\.hf must be true/
  );
  assert.throws(
    () => assertPromotionReady(promotionReadyEntry({
      lifecycle: {
        availability: {
          hf: true,
        },
        status: {
          runtime: 'blocked',
          tested: 'verified',
        },
        tested: {
          contracts: {
            executionContractOk: true,
          },
        },
      },
    })),
    /lifecycle\.status\.runtime must be "active"/
  );
  assert.throws(
    () => assertPromotionReady(promotionReadyEntry({
      lifecycle: {
        availability: {
          hf: true,
        },
        status: {
          runtime: 'active',
          tested: 'verified',
        },
        tested: {
          contracts: {},
        },
      },
    })),
    /execution contract gate must be explicitly true/
  );
  assert.throws(
    () => assertPromotionReady(promotionReadyEntry({ artifactCompleteness: 'incomplete' })),
    /artifactCompleteness must be "complete"/
  );
  assert.throws(
    () => assertPromotionReady(promotionReadyEntry({ weightsRefAllowed: true })),
    /weightsRefAllowed must be false for complete artifact publication/
  );
  assert.doesNotThrow(
    () => assertPromotionReady(promotionReadyEntry({ weightsRefAllowed: true }), { manifestOnly: true })
  );
  assert.throws(
    () => assertPromotionReady(promotionReadyEntry(), { manifestOnly: true }),
    /weightsRefAllowed must be true for --manifest-only weightsRef publication/
  );
}

{
  // Bootstrap requires availability.hf=false (other gates still apply).
  const bootstrapEntry = promotionReadyEntry({
    modelId: 'new-model-first-publish',
    lifecycle: {
      availability: { hf: false },
      status: { runtime: 'active', tested: 'verified' },
      tested: { contracts: { executionContractOk: true } },
    },
  });
  assert.doesNotThrow(() => assertPromotionReady(bootstrapEntry, { bootstrap: true }));
  assert.throws(
    () => assertPromotionReady(bootstrapEntry),
    /lifecycle\.availability\.hf must be true before publication/
  );
  assert.throws(
    () => assertPromotionReady({
      ...bootstrapEntry,
      lifecycle: { ...bootstrapEntry.lifecycle, availability: { hf: true } },
    }, { bootstrap: true }),
    /--bootstrap requires lifecycle\.availability\.hf=false/
  );
  assert.throws(
    () => assertPromotionReady({
      ...bootstrapEntry,
      lifecycle: {
        ...bootstrapEntry.lifecycle,
        status: { runtime: 'blocked', tested: 'verified' },
      },
    }, { bootstrap: true }),
    /lifecycle\.status\.runtime must be "active"/
  );
}

{
  // Dry-run: complete artifact path is echoed, bootstrap is preserved.
  const logs = [];
  const origLog = console.log;
  console.log = (line) => logs.push(String(line));
  try {
    await main([
      '--model-id', 'gemma-4-e2b-it-q4k-ehf16-af32',
      '--dry-run',
    ]);
  } finally {
    console.log = origLog;
  }
  const parsed = JSON.parse(logs[0]);
  assert.equal(parsed.modelId, 'gemma-4-e2b-it-q4k-ehf16-af32');
  assert.equal(parsed.manifestOnly, false);
  assert.equal(parsed.bootstrap, false);
  assert.ok(parsed.shardDir, 'complete-artifact dry-run should include shardDir');
  assert.ok(parsed.manifestDir.endsWith('/models/local/gemma-4-e2b-it-q4k-ehf16-af32'));
}

{
  // writeBackLocalCatalog: standard republish path updates revision in place.
  const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-catalog-test-'));
  const catalogFile = path.join(tmpDir, 'catalog.json');
  const seed = {
    version: 1,
    updatedAt: '2026-04-18',
    models: [
      {
        modelId: 'alpha',
        hf: { repoId: 'Clocksmith/rdrr', revision: 'OLD', path: 'models/alpha' },
        lifecycle: { availability: { hf: true }, status: { runtime: 'active', tested: 'verified' } },
      },
      {
        modelId: 'beta',
        hf: { repoId: 'Clocksmith/rdrr', revision: 'BETA_OLD', path: 'models/beta' },
      },
    ],
  };
  await writeFile(catalogFile, JSON.stringify(seed, null, 2) + '\n', 'utf8');
  try {
    await writeBackLocalCatalog(catalogFile, 'alpha', 'deadbeef');
    const written = JSON.parse(await readFile(catalogFile, 'utf8'));
    assert.equal(written.models[0].hf.revision, 'deadbeef');
    assert.equal(written.models[0].hf.repoId, 'Clocksmith/rdrr');
    assert.equal(written.models[0].hf.path, 'models/alpha');
    assert.equal(written.models[0].lifecycle.availability.hf, true);
    assert.equal(written.models[1].hf.revision, 'BETA_OLD', 'other entries unchanged');

    // Bootstrap flips availability.hf to true.
    await writeBackLocalCatalog(catalogFile, 'beta', 'cafebabe', { bootstrap: true });
    const afterBootstrap = JSON.parse(await readFile(catalogFile, 'utf8'));
    assert.equal(afterBootstrap.models[1].hf.revision, 'cafebabe');
    assert.equal(afterBootstrap.models[1].lifecycle.availability.hf, true);
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

{
  // writeBackLocalCatalog fails fast on missing entry or missing hf.{repoId,path}.
  const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-catalog-fail-'));
  const catalogFile = path.join(tmpDir, 'catalog.json');
  const seed = { version: 1, models: [{ modelId: 'alpha' }] };
  await writeFile(catalogFile, JSON.stringify(seed, null, 2), 'utf8');
  try {
    await assert.rejects(
      writeBackLocalCatalog(catalogFile, 'missing', 'abc123'),
      /not found.*during catalog writeback/
    );
    await assert.rejects(
      writeBackLocalCatalog(catalogFile, 'alpha', 'abc123'),
      /missing hf\.repoId or hf\.path/
    );
    await assert.rejects(
      writeBackLocalCatalog(catalogFile, 'alpha', ''),
      /requires a non-empty revision/
    );
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

console.log('publish-hf-registry-model.test: ok');
