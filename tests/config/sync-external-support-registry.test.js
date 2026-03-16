import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import {
  buildExternalSupportRegistry,
  main as syncExternalSupportRegistryMain,
} from '../../tools/sync-external-support-registry.js';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-external-support-registry-'));

try {
  const rdrrIndexPath = path.join(root, 'RDRR_INDEX.json');
  const catalogPath = path.join(root, 'catalog.json');
  mkdirSync(path.join(root, 'rdrr'), { recursive: true });
  mkdirSync(path.join(root, 'rdrr', 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16'), { recursive: true });
  mkdirSync(path.join(root, 'rdrr', 'gemma-3-1b-it-f16-af32'), { recursive: true });

  writeFileSync(
    path.join(root, 'rdrr', 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16', 'manifest.json'),
    JSON.stringify({
      modelId: 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16',
      shards: [{ filename: 'shard_00000.bin' }],
    }, null, 2),
    'utf8'
  );
  writeFileSync(
    path.join(root, 'rdrr', 'gemma-3-1b-it-f16-af32', 'manifest.json'),
    JSON.stringify({
      modelId: 'gemma-3-1b-it-f16-af32',
      shards: [{ filename: 'shard_00000.bin' }],
    }, null, 2),
    'utf8'
  );

  writeFileSync(rdrrIndexPath, JSON.stringify({
    schemaVersion: 1,
    generatedAt: '2026-03-11T00:00:00.000Z',
    volumeRoot: root,
    rdrrRoot: path.join(root, 'rdrr'),
    sourceModels: [
      {
        sourceModel: 'Qwen/Qwen3.5-0.8B',
        sourceFormats: ['safetensors'],
        variantCount: 1,
        variants: [
          {
            rdrrModelId: 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16',
            variant: 'wq4k-ef16-hf16-f16',
            quantization: 'Q4_K_M',
            convertedAt: '2026-03-05T14:59:04.365Z',
            totalSizeBytes: 788833984,
            shardCount: 12,
            sourceFormat: 'safetensors',
            sourceRevision: '2fc06364715b967f1860aea9cf38778875588b17',
            pathRelativeToVolume: 'rdrr/qwen-3-5-0-8b-wq4k-ef16-hf16-f16',
            pathRelativeToRdrrRoot: 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16',
            manifestPath: `${root}/rdrr/qwen-3-5-0-8b-wq4k-ef16-hf16-f16/manifest.json`,
            hasOrigin: true,
          },
        ],
      },
      {
        sourceModel: 'google/gemma-3-1b-it',
        sourceFormats: ['safetensors'],
        variantCount: 1,
        variants: [
          {
            rdrrModelId: 'gemma-3-1b-it-f16-af32',
            variant: 'f16-af32',
            quantization: 'F16',
            convertedAt: '2026-03-10T16:27:00.122Z',
            totalSizeBytes: 1999771904,
            shardCount: 30,
            sourceFormat: 'safetensors',
            sourceRevision: 'dcc83ea841ab6100d6b47a070329e1ba4cf78752',
            pathRelativeToVolume: 'rdrr/gemma-3-1b-it-f16-af32',
            pathRelativeToRdrrRoot: 'gemma-3-1b-it-f16-af32',
            manifestPath: `${root}/rdrr/gemma-3-1b-it-f16-af32/manifest.json`,
            hasOrigin: true,
          },
        ],
      },
    ],
  }, null, 2), 'utf8');

  writeFileSync(catalogPath, JSON.stringify({
    version: 1,
    lifecycleSchemaVersion: 1,
    updatedAt: '2026-03-10',
    models: [
      {
        modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
        aliases: [
          'qwen-3-5-0-8b',
          'qwen-3-5-0-8b-wq4k-ef16-hf16-f16',
        ],
        modes: ['run'],
        artifact: {
          format: 'rdrr',
        },
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
          path: 'models/qwen-3-5-0-8b-q4k-ehaf16',
        },
        lifecycle: {
          availability: {
            hf: true,
            local: false,
            curated: false,
          },
          status: {
            runtime: 'active',
            conversion: 'ready',
            demo: 'none',
            tested: 'failing',
          },
        },
      },
      {
        modelId: 'gemma-3-1b-it-f16-af32',
        aliases: [],
        modes: ['run'],
        artifact: {
          format: 'rdrr',
        },
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: 'beadbeadbeadbeadbeadbeadbeadbeadbeadbead',
          path: 'models/gemma-3-1b-it-f16-af32',
        },
        lifecycle: {
          availability: {
            hf: true,
            local: true,
            curated: false,
          },
          status: {
            runtime: 'active',
            conversion: 'ready',
            demo: 'none',
            tested: 'verified',
          },
        },
      },
    ],
  }, null, 2), 'utf8');

  const outputs = await buildExternalSupportRegistry({
    volumeRoot: root,
    rdrrIndex: rdrrIndexPath,
    catalogFile: catalogPath,
    jsonOutput: path.join(root, 'DOPPLER_SUPPORT_REGISTRY.json'),
    mdOutput: path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'),
  }, '2026-03-11T12:00:00.000Z');

  const payload = JSON.parse(outputs.json);
  assert.equal(payload.supportSource, catalogPath);
  assert.equal(payload.summary.catalogModelCount, 2);
  assert.equal(payload.summary.catalogBackedRdrrCount, 2);
  assert.equal(payload.summary.verifiedCount, 1);
  assert.equal(payload.summary.hfApprovedCount, 2);
  assert.equal(payload.summary.hfPathMismatchCount, 1);
  assert.equal(payload.summary.manifestModelIdMismatchCount, 1);
  assert.equal(payload.summary.uncatalogedRdrrCount, 0);

  const qwen = payload.models.find((entry) => entry.modelId === 'qwen-3-5-0-8b-q4k-ehaf16');
  assert.ok(qwen);
  assert.equal(qwen.external.rdrrModelId, 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16');
  assert.equal(qwen.external.manifestModelId, 'qwen-3-5-0-8b-wq4k-ef16-hf16-f16');
  assert.equal(qwen.external.manifestModelIdMatchesCatalogModelId, false);
  assert.equal(qwen.external.hostedPathMatchesRdrr, false);

  const gemma = payload.models.find((entry) => entry.modelId === 'gemma-3-1b-it-f16-af32');
  assert.ok(gemma);
  assert.equal(gemma.external.rdrrModelId, 'gemma-3-1b-it-f16-af32');
  assert.equal(gemma.external.manifestModelId, 'gemma-3-1b-it-f16-af32');
  assert.equal(gemma.external.manifestModelIdMatchesCatalogModelId, true);
  assert.equal(gemma.external.hostedPathMatchesRdrr, true);

  writeFileSync(path.join(root, 'DOPPLER_SUPPORT_REGISTRY.json'), outputs.json, 'utf8');
  writeFileSync(path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'), outputs.md, 'utf8');
  await syncExternalSupportRegistryMain([
    '--check',
    '--volume-root', root,
    '--rdrr-index', rdrrIndexPath,
    '--catalog-file', catalogPath,
    '--source-support-file', catalogPath,
    '--json-output', path.join(root, 'DOPPLER_SUPPORT_REGISTRY.json'),
    '--md-output', path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'),
  ]);

  const supportRegistryPath = path.join(root, 'DOPPLER_SUPPORT_REGISTRY.json');
  writeFileSync(supportRegistryPath, JSON.stringify({
    version: 1,
    lifecycleSchemaVersion: 1,
    updatedAt: '2026-03-11',
    models: [
      {
        modelId: 'gemma-3-1b-it-f16-af32',
        aliases: [],
        modes: ['run'],
        artifact: {
          format: 'rdrr',
        },
        hf: {
          repoId: 'Clocksmith/rdrr',
          revision: 'feedfeedfeedfeedfeedfeedfeedfeedfeedfeed',
          path: 'models/gemma-3-1b-it-f16-af32',
        },
        lifecycle: {
          availability: {
            hf: true,
            local: true,
            curated: false,
          },
          status: {
            runtime: 'active',
            conversion: 'ready',
            demo: 'none',
            tested: 'verified',
          },
        },
      },
    ],
  }, null, 2), 'utf8');

  const canonicalOutputs = await buildExternalSupportRegistry({
    volumeRoot: root,
    rdrrIndex: rdrrIndexPath,
    catalogFile: catalogPath,
    sourceSupportFile: supportRegistryPath,
    jsonOutput: supportRegistryPath,
    mdOutput: path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'),
  }, '2026-03-11T12:30:00.000Z');

  const canonicalPayload = JSON.parse(canonicalOutputs.json);
  assert.equal(canonicalPayload.supportSource, supportRegistryPath);
  assert.deepEqual(canonicalPayload.models.map((entry) => entry.modelId), ['gemma-3-1b-it-f16-af32']);

  await assert.rejects(
    () => buildExternalSupportRegistry({
      volumeRoot: root,
      rdrrIndex: rdrrIndexPath,
      catalogFile: catalogPath,
      jsonOutput: supportRegistryPath,
      mdOutput: path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'),
    }, '2026-03-11T12:45:00.000Z'),
    /Canonical external support registry is missing promotable RDRR-backed repo entries: qwen-3-5-0-8b-q4k-ehaf16/
  );

  const promotedOutputs = await buildExternalSupportRegistry({
    volumeRoot: root,
    rdrrIndex: rdrrIndexPath,
    catalogFile: catalogPath,
    sourceSupportFile: catalogPath,
    jsonOutput: supportRegistryPath,
    mdOutput: path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'),
  }, '2026-03-11T13:00:00.000Z');

  const promotedPayload = JSON.parse(promotedOutputs.json);
  assert.equal(promotedPayload.supportSource, catalogPath);
  assert.deepEqual(
    promotedPayload.models.map((entry) => entry.modelId),
    ['qwen-3-5-0-8b-q4k-ehaf16', 'gemma-3-1b-it-f16-af32']
  );

  writeFileSync(supportRegistryPath, promotedOutputs.json, 'utf8');
  writeFileSync(path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'), promotedOutputs.md, 'utf8');
  await syncExternalSupportRegistryMain([
    '--check',
    '--volume-root', root,
    '--rdrr-index', rdrrIndexPath,
    '--catalog-file', catalogPath,
    '--json-output', supportRegistryPath,
    '--md-output', path.join(root, 'DOPPLER_SUPPORT_REGISTRY.md'),
  ]);
}
finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('sync-external-support-registry.test: ok');
