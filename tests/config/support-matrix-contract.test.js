import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import {
  buildCurrentInferenceStatusBuckets,
  parseArgs,
  resolveRowStatus,
  validateCatalogMatrixInputs,
  validateGemma4EvidenceFiles,
  validateGemma4TargetMatrixInputs,
} from '../../tools/sync-model-support-matrix.js';
import { assertManifestArtifactIntegrity } from '../helpers/local-model-fixture.js';

{
  assert.deepEqual(validateCatalogMatrixInputs({
    updatedAt: '2026-03-06',
    models: [
      {
        modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
        family: 'gemma3',
        classification: {
          domain: 'language',
          tasks: ['generation'],
          architectureRole: 'autoregressive-decoder',
          inputs: ['text'],
          outputs: ['text'],
        },
        artifact: {
          format: 'rdrr',
        },
        baseUrl: './local/gemma-3-270m-it-q4k-ehf16-af32',
        hf: {
          repoId: 'clocksmith/rdrr',
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
          repoId: 'clocksmith/rdrr',
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
    'broken-model: classification must be an object',
    'broken-model: classification must be an object',
  ]);
}

{
  const catalogModels = [
    {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      family: 'gemma4',
      lifecycle: {
        status: {
          tested: 'verified',
        },
      },
    },
  ];
  const quickstartModels = [
    {
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      modes: ['text', 'vision'],
    },
  ];

  assert.deepEqual(validateGemma4TargetMatrixInputs({
    schemaVersion: 1,
    sourceUrls: [
      'https://ai.google.dev/gemma/docs/core',
      'https://ai.google.dev/gemma/docs/releases',
      'https://developers.google.com/edge/litert-lm/models/gemma-4',
    ],
    targets: [
      {
        targetId: 'gemma-4-e2b',
        officialName: 'Gemma 4 E2B',
        dopplerStatus: 'partially_verified',
        surfaceStatus: {
          browser: 'unverified',
          electron: 'unverified',
          node: 'verified',
        },
        currentLanes: [
          {
            modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
            role: 'complete-q4k-text-vision-weight-pack',
            claimStatus: 'verified',
          },
        ],
        serveStatus: 'unverified',
        servedLanes: [
          'gemma-4-e2b-it-q4k-ehf16-af32',
        ],
        evidence: {
          runtimeReceipts: [
            {
              modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
              surface: 'node',
              status: 'pass',
              path: 'reports/gemma-4-e2b-it-q4k-ehf16-af32/2026-05-07T14-35-06.195Z.json',
            },
          ],
          benchmarkReceipts: [
            {
              modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
              surface: 'browser',
              status: 'performance_evidence',
              path: 'benchmarks/vendors/results/compare_20260421T001902.json',
            },
          ],
          serveReceipts: [],
          preflightReceipts: [],
        },
        missing: [
          'doppler-serve runtime pass receipt',
          'electron receipt',
          'mtp lane',
        ],
        blockers: [
          {
            code: 'browser-runtime-pass-receipt-missing',
            surface: 'browser',
            state: 'unverified',
            reason: 'Browser benchmark evidence exists, but no browser runtime pass receipt is listed for this target.',
          },
          {
            code: 'serve-runtime-pass-receipt-missing',
            surface: 'serve',
            state: 'unverified',
            reason: 'E2B lanes are listed in the doppler-serve registry, but no doppler-serve runtime pass receipt is listed for this target.',
          },
          {
            code: 'electron-receipt-missing',
            surface: 'electron',
            state: 'unverified',
            reason: 'No committed Electron runtime or benchmark receipt is listed for this target.',
          },
          {
            code: 'mtp-lane-not-implemented',
            surface: 'mtp',
            state: 'not_implemented',
            reason: 'Official Gemma 4 MTP exists, but Doppler has no MTP execution lane or receipt for this target.',
          },
        ],
        officialMtp: true,
        mtpStatus: 'not_implemented',
      },
      {
        targetId: 'gemma-4-e4b',
        officialName: 'Gemma 4 E4B',
        dopplerStatus: 'gap',
        surfaceStatus: {
          browser: 'unsupported',
          electron: 'unsupported',
          node: 'unsupported',
        },
        currentLanes: [],
        serveStatus: 'unsupported',
        servedLanes: [],
        evidence: {
          runtimeReceipts: [],
          benchmarkReceipts: [],
          serveReceipts: [],
          preflightReceipts: [],
        },
        missing: [
          'source package profile',
          'mtp lane',
        ],
        blockers: [
          {
            code: 'browser-runtime-unsupported',
            surface: 'browser',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 E4B lane is listed for browser execution.',
          },
          {
            code: 'electron-runtime-unsupported',
            surface: 'electron',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 E4B lane is listed for Electron execution.',
          },
          {
            code: 'node-runtime-unsupported',
            surface: 'node',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 E4B lane is listed for Node execution.',
          },
          {
            code: 'serve-runtime-unsupported',
            surface: 'serve',
            state: 'unsupported',
            reason: 'No doppler-serve lane is available because no current lane is listed.',
          },
          {
            code: 'source-package-profile-missing',
            surface: 'model',
            state: 'missing',
            reason: 'No source-package profile is listed.',
          },
          {
            code: 'mtp-lane-not-implemented',
            surface: 'mtp',
            state: 'not_implemented',
            reason: 'Official Gemma 4 MTP exists, but Doppler has no MTP execution lane or receipt for this target.',
          },
        ],
        officialMtp: true,
        mtpStatus: 'not_implemented',
      },
      {
        targetId: 'gemma-4-12b-unified',
        officialName: 'Gemma 4 12B Unified',
        dopplerStatus: 'gap',
        surfaceStatus: {
          browser: 'unsupported',
          electron: 'unsupported',
          node: 'unsupported',
        },
        currentLanes: [],
        serveStatus: 'unsupported',
        servedLanes: [],
        evidence: {
          runtimeReceipts: [],
          benchmarkReceipts: [],
          serveReceipts: [],
          preflightReceipts: [],
        },
        missing: [
          'browser receipt',
          'mtp lane',
        ],
        blockers: [
          {
            code: 'browser-runtime-unsupported',
            surface: 'browser',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 12B Unified lane is listed for browser execution.',
          },
          {
            code: 'electron-runtime-unsupported',
            surface: 'electron',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 12B Unified lane is listed for Electron execution.',
          },
          {
            code: 'node-runtime-unsupported',
            surface: 'node',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 12B Unified lane is listed for Node execution.',
          },
          {
            code: 'serve-runtime-unsupported',
            surface: 'serve',
            state: 'unsupported',
            reason: 'No doppler-serve lane is available because no current lane is listed.',
          },
          {
            code: 'browser-receipt-missing',
            surface: 'model',
            state: 'missing',
            reason: 'No browser receipt is listed.',
          },
          {
            code: 'mtp-lane-not-implemented',
            surface: 'mtp',
            state: 'not_implemented',
            reason: 'Official Gemma 4 MTP exists, but Doppler has no MTP execution lane or receipt for this target.',
          },
        ],
        officialMtp: true,
        mtpStatus: 'not_implemented',
      },
      {
        targetId: 'gemma-4-31b',
        officialName: 'Gemma 4 31B',
        dopplerStatus: 'gap',
        surfaceStatus: {
          browser: 'unsupported',
          electron: 'unsupported',
          node: 'unsupported',
        },
        currentLanes: [],
        serveStatus: 'unsupported',
        servedLanes: [],
        evidence: {
          runtimeReceipts: [],
          benchmarkReceipts: [],
          serveReceipts: [],
          preflightReceipts: [],
        },
        missing: [
          'electron receipt',
          'mtp lane',
        ],
        blockers: [
          {
            code: 'browser-runtime-unsupported',
            surface: 'browser',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 31B lane is listed for browser execution.',
          },
          {
            code: 'electron-runtime-unsupported',
            surface: 'electron',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 31B lane is listed for Electron execution.',
          },
          {
            code: 'node-runtime-unsupported',
            surface: 'node',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 31B lane is listed for Node execution.',
          },
          {
            code: 'serve-runtime-unsupported',
            surface: 'serve',
            state: 'unsupported',
            reason: 'No doppler-serve lane is available because no current lane is listed.',
          },
          {
            code: 'electron-receipt-missing',
            surface: 'model',
            state: 'missing',
            reason: 'No Electron receipt is listed.',
          },
          {
            code: 'mtp-lane-not-implemented',
            surface: 'mtp',
            state: 'not_implemented',
            reason: 'Official Gemma 4 MTP exists, but Doppler has no MTP execution lane or receipt for this target.',
          },
        ],
        officialMtp: true,
        mtpStatus: 'not_implemented',
      },
      {
        targetId: 'gemma-4-26b-a4b',
        officialName: 'Gemma 4 26B A4B',
        dopplerStatus: 'gap',
        surfaceStatus: {
          browser: 'unsupported',
          electron: 'unsupported',
          node: 'unsupported',
        },
        currentLanes: [],
        serveStatus: 'unsupported',
        servedLanes: [],
        evidence: {
          runtimeReceipts: [],
          benchmarkReceipts: [],
          serveReceipts: [],
          preflightReceipts: [],
        },
        missing: [
          'source package profile',
          'mtp lane',
        ],
        blockers: [
          {
            code: 'browser-runtime-unsupported',
            surface: 'browser',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 26B A4B lane is listed for browser execution.',
          },
          {
            code: 'electron-runtime-unsupported',
            surface: 'electron',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 26B A4B lane is listed for Electron execution.',
          },
          {
            code: 'node-runtime-unsupported',
            surface: 'node',
            state: 'unsupported',
            reason: 'No Doppler Gemma 4 26B A4B lane is listed for Node execution.',
          },
          {
            code: 'serve-runtime-unsupported',
            surface: 'serve',
            state: 'unsupported',
            reason: 'No doppler-serve lane is available because no current lane is listed.',
          },
          {
            code: 'source-package-profile-missing',
            surface: 'model',
            state: 'missing',
            reason: 'No source-package profile is listed.',
          },
          {
            code: 'mtp-lane-not-implemented',
            surface: 'mtp',
            state: 'not_implemented',
            reason: 'Official Gemma 4 MTP exists, but Doppler has no MTP execution lane or receipt for this target.',
          },
        ],
        officialMtp: true,
        mtpStatus: 'not_implemented',
      },
    ],
  }, catalogModels, quickstartModels), []);

  assert.deepEqual(validateGemma4TargetMatrixInputs({
    schemaVersion: 1,
    targets: [
      {
        targetId: 'gemma-4-e4b',
        dopplerStatus: 'gap',
        surfaceStatus: {
          browser: 'verified',
          electron: 'unsupported',
          node: 'unsupported',
        },
        currentLanes: [
          {
            modelId: 'missing-model',
            claimStatus: 'verified',
          },
        ],
        mtpStatus: 'unknown',
      },
    ],
  }, catalogModels, quickstartModels), [
    'Gemma 4 target matrix sourceUrls must include https://ai.google.dev/gemma/docs/core',
    'Gemma 4 target matrix sourceUrls must include https://ai.google.dev/gemma/docs/releases',
    'Gemma 4 target matrix sourceUrls must include https://developers.google.com/edge/litert-lm/models/gemma-4',
    'gemma-4-e4b: officialName is required',
    'gemma-4-e4b: invalid mtpStatus',
    'gemma-4-e4b: officialMtp must be true',
    'gemma-4-e4b: evidence is required',
    'gemma-4-e4b: evidence.runtimeReceipts must be an array',
    'gemma-4-e4b: evidence.benchmarkReceipts must be an array',
    'gemma-4-e4b: evidence.serveReceipts must be an array',
    'gemma-4-e4b: evidence.preflightReceipts must be an array',
    'gemma-4-e4b: invalid serveStatus',
    'gemma-4-e4b: servedLanes must be an array',
    'gemma-4-e4b: missing must be an array',
    'gemma-4-e4b: blockers must be an array',
    'gemma-4-e4b: gap targets must not list current lanes',
    'gemma-4-e4b: lane missing-model role is required',
    'gemma-4-e4b: lane missing-model is missing from models/catalog.json',
    'gemma-4-e4b: verified lane missing-model must have passing receipt evidence',
    'gemma-4-e4b: verified browser surface must have same-surface runtime pass evidence',
    'missing Gemma 4 targetId: gemma-4-e2b',
    'missing Gemma 4 targetId: gemma-4-12b-unified',
    'missing Gemma 4 targetId: gemma-4-31b',
    'missing Gemma 4 targetId: gemma-4-26b-a4b',
  ]);
}

{
  const matrix = JSON.parse(await fs.readFile('models/gemma4-targets.json', 'utf8'));
  assert.deepEqual(await validateGemma4EvidenceFiles(matrix), []);

  const errors = await validateGemma4EvidenceFiles({
    targets: [
      {
        targetId: 'gemma-4-e2b',
        evidence: {
          runtimeReceipts: [
            {
              modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
              status: 'pass',
              path: 'package.json',
            },
          ],
          benchmarkReceipts: [
            {
              modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
              status: 'performance_evidence',
              path: 'package.json',
            },
          ],
        },
      },
    ],
  });

  assert.ok(errors.includes(
    'gemma-4-e2b: runtime receipt package.json modelId mismatch (unknown-model != gemma-4-e2b-it-q4k-ehf16-af32)'
  ));
  assert.ok(errors.includes(
    'gemma-4-e2b: benchmark receipt package.json missing numeric timing.decodeTokensPerSec'
  ));
  assert.ok(errors.includes(
    'gemma-4-e2b: benchmark receipt package.json missing memoryStats.kvCache'
  ));
  assert.ok(errors.includes(
    'gemma-4-e2b: benchmark receipt package.json performance evidence requires decode validity'
  ));
}

{
  const repoRoot = process.cwd();
  const tempDir = path.join(repoRoot, '.tmp', `support-matrix-preflight-${process.pid}`);
  const receiptPath = path.join(tempDir, 'preflight-receipt.json');
  const receiptRelPath = path.relative(repoRoot, receiptPath).replace(/\\/g, '/');
  const sourceReceipt = JSON.parse(await fs.readFile(
    'reports/gemma4-preflight/gemma-4-e2b-it-q4k-ehf16-af32/2026-06-20T201732Z.preflight.json',
    'utf8'
  ));
  sourceReceipt.manifest.sha256 = `sha256:${'0'.repeat(64)}`;
  await fs.mkdir(tempDir, { recursive: true });
  try {
    await fs.writeFile(receiptPath, `${JSON.stringify(sourceReceipt, null, 2)}\n`, 'utf8');
    const errors = await validateGemma4EvidenceFiles({
      targets: [
        {
          targetId: 'gemma-4-e2b',
          evidence: {
            runtimeReceipts: [],
            benchmarkReceipts: [],
            serveReceipts: [],
            preflightReceipts: [
              {
                modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
                surface: 'node',
                status: 'pass',
                path: receiptRelPath,
              },
            ],
          },
        },
      ],
    });
    assert.ok(errors.some((error) => error.includes('manifest sha256 mismatch')));
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

{
  const repoRoot = process.cwd();
  const tempDir = path.join(repoRoot, '.tmp', `support-matrix-serve-${process.pid}`);
  const receiptPath = path.join(tempDir, 'serve-receipt.json');
  const diagnosticReceiptPath = path.join(tempDir, 'serve-diagnostic-receipt.json');
  const receiptRelPath = path.relative(repoRoot, receiptPath).replace(/\\/g, '/');
  const diagnosticReceiptRelPath = path.relative(repoRoot, diagnosticReceiptPath).replace(/\\/g, '/');
  await fs.mkdir(tempDir, { recursive: true });
  try {
    await fs.writeFile(receiptPath, `${JSON.stringify({
      receiptVersion: 'doppler_serve_receipt_v1',
      schemaVersion: 1,
      surface: 'serve',
      status: 'pass',
      runtime: 'doppler-gpu',
      runtimePath: 'doppler-gpu.chatText',
      runtimeModelSource: {
        kind: 'quickstart-registry',
        modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      },
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      resolvedModel: 'gemma-4-e2b-it-q4k-ehf16-af32',
      artifact: {
        sourceCheckpointId: 'google/gemma-4-e2b-it',
        weightPackId: 'gemma-4-e2b-it-q4k-ehf16-af32-wp-catalog-v1',
        manifestVariantId: 'gemma-4-e2b-it-q4k-ehf16-af32-mv-exec-v1',
      },
      request: {
        messages: {
          count: 1,
          digest: {
            algorithm: 'sha256',
            value: 'a'.repeat(64),
            bytes: 42,
          },
        },
        generationDigest: {
          algorithm: 'sha256',
          value: 'b'.repeat(64),
          bytes: 64,
        },
      },
      output: {
        role: 'assistant',
        digest: {
          algorithm: 'sha256',
          value: 'c'.repeat(64),
          bytes: 19,
        },
        textLength: 19,
        empty: false,
      },
      transcript: {
        digest: {
          algorithm: 'sha256',
          value: 'd'.repeat(64),
          bytes: 160,
        },
      },
      usage: {
        promptTokens: 4,
        completionTokens: 3,
        totalTokens: 7,
      },
    }, null, 2)}\n`, 'utf8');

    assert.deepEqual(await validateGemma4EvidenceFiles({
      targets: [
        {
          targetId: 'gemma-4-e2b',
          evidence: {
            runtimeReceipts: [],
            benchmarkReceipts: [],
            serveReceipts: [
              {
                modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
                surface: 'serve',
                status: 'pass',
                path: receiptRelPath,
              },
            ],
          },
        },
      ],
    }), []);

    await fs.writeFile(diagnosticReceiptPath, `${JSON.stringify({
      receiptVersion: 'doppler_serve_receipt_v1',
      schemaVersion: 1,
      surface: 'serve',
      status: 'diagnostic',
      runtime: 'doppler-gpu',
      runtimePath: 'doppler-gpu.chatText',
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      resolvedModel: 'gemma-4-e2b-it-q4k-ehf16-af32',
      artifact: {
        sourceCheckpointId: 'google/gemma-4-e2b-it',
        weightPackId: 'gemma-4-e2b-it-q4k-ehf16-af32-wp-catalog-v1',
        manifestVariantId: 'gemma-4-e2b-it-q4k-ehf16-af32-mv-exec-v1',
      },
      request: {
        messages: {
          count: 1,
          digest: {
            algorithm: 'sha256',
            value: 'a'.repeat(64),
            bytes: 42,
          },
        },
        generationDigest: {
          algorithm: 'sha256',
          value: 'b'.repeat(64),
          bytes: 64,
        },
      },
      generation: {
        maxTokens: 8,
        temperature: 0,
        topP: null,
        topK: 1,
      },
      failure: {
        code: 'pipeline-load-failed',
        stage: 'loadWeights',
        message: 'Storage buffer size exceeds maxStorageBufferBindingSize.',
        modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        weightLoadFailure: {
          tensorName: 'model.language_model.embed_tokens.weight',
          tensorRole: 'embedding',
          tensorDtype: 'F16',
          tensorShape: [262144, 1536],
          tensorSizeBytes: 805306368,
          tensorLoadStage: 'gpuResidentEmbeddingLimitPreflight',
          toGPU: true,
          streamedUpload: false,
          deviceLimitFailure: {
            kind: 'gpu_resident_embedding_exceeds_device_limit',
            maxGpuResidentBytes: 134217728,
            maxStorageBufferBindingSize: 134217728,
            maxBufferSize: 4294967295,
            maxStorageBuffersPerShaderStage: null,
            largeWeightMaxBytes: 120795955,
            embeddingKernel: {
              kernel: 'gather.wgsl',
              entry: 'main',
            },
            splitKernelExpected: false,
            activeSplitKernelMaxSections: null,
            maxSplitEmbeddingSections: 8,
            requiredSplitSections: 7,
          },
        },
      },
    }, null, 2)}\n`, 'utf8');

    assert.deepEqual(await validateGemma4EvidenceFiles({
      targets: [
        {
          targetId: 'gemma-4-e2b',
          evidence: {
            runtimeReceipts: [],
            benchmarkReceipts: [],
            serveReceipts: [
              {
                modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
                surface: 'serve',
                status: 'diagnostic',
                path: diagnosticReceiptRelPath,
              },
            ],
          },
        },
      ],
    }), []);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

{
  const repoRoot = process.cwd();
  const catalogPath = path.join(repoRoot, 'models', 'catalog.json');
  const catalog = JSON.parse(await fs.readFile(catalogPath, 'utf8'));
  assert.deepEqual(validateCatalogMatrixInputs(catalog), []);
  const noteReceiptPathPattern = /(?:reports|benchmarks\/vendors\/results)\/[^\s),;]+\.json/g;

  for (const entry of catalog.models) {
    const testedNotes = typeof entry?.lifecycle?.tested?.notes === 'string'
      ? entry.lifecycle.tested.notes
      : '';
    for (const receiptPath of testedNotes.match(noteReceiptPathPattern) ?? []) {
      assert.ok(
        existsSync(path.join(repoRoot, receiptPath)),
        `${entry.modelId}: catalog note receipt path must exist: ${receiptPath}`
      );
    }
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
  const textGeneratorClassification = {
    domain: 'language',
    tasks: ['generation'],
    architectureRole: 'autoregressive-decoder',
    inputs: ['text'],
    outputs: ['text'],
  };
  const buckets = buildCurrentInferenceStatusBuckets({
    catalogModels: [
      {
        modelId: 'verified-model',
        family: 'gemma3',
        classification: textGeneratorClassification,
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
        classification: textGeneratorClassification,
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
        classification: textGeneratorClassification,
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
