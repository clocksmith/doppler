import assert from 'node:assert/strict';
import {
  PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID,
  PROGRAM_BUNDLE_HOST_JS_SUBSET,
  PROGRAM_BUNDLE_HOST_SCHEMA_ID,
  PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
  PROGRAM_BUNDLE_SCHEMA_ID,
  PROGRAM_BUNDLE_SCHEMA_VERSION,
  validateProgramBundle,
} from '../../src/config/schema/program-bundle.schema.js';

const DIGEST_A = `sha256:${'a'.repeat(64)}`;
const DIGEST_B = `sha256:${'b'.repeat(64)}`;
const DIGEST_C = `sha256:${'c'.repeat(64)}`;
const DIGEST_D = `sha256:${'d'.repeat(64)}`;
const DIGEST_E = `sha256:${'e'.repeat(64)}`;
const DIGEST_F = `sha256:${'f'.repeat(64)}`;

function createBundle() {
  return {
    schema: PROGRAM_BUNDLE_SCHEMA_ID,
    schemaVersion: PROGRAM_BUNDLE_SCHEMA_VERSION,
    bundleId: 'unit-bundle',
    modelId: 'unit-model',
    createdAtUtc: '2026-04-22T00:00:00.000Z',
    sources: {
      manifest: { path: 'models/unit/manifest.json', hash: DIGEST_A },
      conversionConfig: null,
      executionGraph: {
        schema: 'doppler.execution/v1',
        hash: DIGEST_B,
        expandedStepHash: DIGEST_C,
      },
      weightSetHash: DIGEST_D,
      artifactSetHash: DIGEST_E,
    },
    host: {
      schema: PROGRAM_BUNDLE_HOST_SCHEMA_ID,
      jsSubset: PROGRAM_BUNDLE_HOST_JS_SUBSET,
      entrypoints: [
        {
          id: 'text-generation',
          module: 'src/inference/pipelines/text/generator.js',
          export: 'PipelineGenerator',
          role: 'model-orchestration',
        },
      ],
      constraints: {
        dynamicImport: 'disallowed',
        dom: 'disallowed-in-model-path',
        filesystem: 'declared-artifacts-only',
        network: 'declared-artifacts-only',
      },
    },
    wgslModules: [
      {
        id: 'embed',
        file: 'gather.wgsl',
        entry: 'main',
        digest: DIGEST_F,
        sourcePath: 'src/gpu/kernels/gather.wgsl',
        reachable: true,
        metadata: {
          entry: 'main',
          sourceMetadataHash: DIGEST_A,
          bindings: [{ group: 0, binding: 0, addressSpace: 'storage', access: 'read', name: 'input' }],
          overrides: [],
          workgroupSize: ['64'],
          requiresSubgroups: false,
        },
      },
    ],
    execution: {
      graphHash: DIGEST_B,
      stepMetadataHash: DIGEST_C,
      kernelClosure: {
        declaredKernelIds: ['embed'],
        reachableKernelIds: ['embed'],
        excludedKernelIds: [],
        undeclaredKernelRefs: [],
        expandedStepCount: 1,
        phases: {
          prefill: 0,
          decode: 0,
          preLayer: 1,
          postLayer: 0,
        },
      },
      steps: [
        {
          id: 'preLayer_both_0_embed',
          index: 0,
          op: 'embed',
          phase: 'both',
          section: 'preLayer',
          layers: 'all',
          src: 'state',
          dst: 'state',
          kernelId: 'embed',
          kernel: 'gather.wgsl',
          entry: 'main',
          kernelDigest: DIGEST_F,
          weights: null,
          constants: null,
          precision: null,
          dispatch: {
            phase: 'both',
            workgroups: 'symbolic:preLayer:both:embed',
            bindings: [{ group: 0, binding: 0, addressSpace: 'storage', access: 'read', name: 'input' }],
          },
        },
      ],
    },
    captureProfile: {
      schema: PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID,
      deterministic: true,
      phases: ['prefill', 'decode'],
      surfaces: ['browser-webgpu'],
      adapter: {
        source: 'reference-report',
      },
      hashPolicy: {
        graph: 'stable-json-sha256',
        dispatch: 'stable-json-sha256',
        transcript: 'stable-json-sha256',
      },
      captureHash: DIGEST_D,
    },
    artifacts: [
      {
        role: 'manifest',
        path: 'models/unit/manifest.json',
        hash: DIGEST_A,
        sizeBytes: 123,
      },
      {
        role: 'reference-report',
        path: 'tests/fixtures/reports/unit/report.json',
        hash: DIGEST_E,
        sizeBytes: 456,
      },
    ],
    referenceTranscript: {
      schema: PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
      source: {
        kind: 'browser-report',
        path: 'tests/fixtures/reports/unit/report.json',
        hash: DIGEST_E,
      },
      executionGraphHash: DIGEST_B,
      surface: 'browser-debug',
      prompt: {
        identity: 'The sky is',
        hash: DIGEST_A,
        tokenIdsHash: null,
        tokenCount: null,
      },
      output: {
        textHash: DIGEST_C,
        tokensGenerated: 1,
        stopReason: 'max-tokens',
        stopTokenId: null,
      },
      tokens: {
        generatedTokenIdsHash: DIGEST_D,
        generatedTextHash: DIGEST_C,
        preview: [{ id: 42, text: ' blue', fallbackText: ' blue' }],
        perStep: [{ index: 0, tokenId: 42, tokenHash: DIGEST_A }],
      },
      phase: {
        prefillMs: 1,
        decodeMs: 2,
        prefillTokens: 3,
        decodeTokens: 1,
      },
      kvCache: {
        mode: 'stats',
        layout: 'contiguous',
        kvDtype: 'f16',
        seqLen: 4,
        maxSeqLen: 32,
        usedBytes: 128,
        allocatedBytes: 1024,
        counters: null,
        stateHash: DIGEST_B,
      },
      logits: {
        mode: 'not-captured',
        reason: 'unit',
        perStepDigests: null,
      },
      tolerance: {
        tokenPolicy: 'exact',
        logitsPolicy: 'not captured',
      },
    },
  };
}

validateProgramBundle(createBundle());

{
  const bundle = createBundle();
  delete bundle.referenceTranscript;
  assert.throws(
    () => validateProgramBundle(bundle),
    /referenceTranscript must be a non-null object/
  );
}

{
  const bundle = createBundle();
  bundle.host.constraints.dynamicImport = 'allowed';
  assert.throws(
    () => validateProgramBundle(bundle),
    /dynamicImport must be "disallowed"/
  );
}

{
  const bundle = createBundle();
  bundle.execution.kernelClosure.reachableKernelIds = ['missing'];
  assert.throws(
    () => validateProgramBundle(bundle),
    /reachable kernel "missing" is missing from wgslModules/
  );
}

console.log('program-bundle-schema.test: ok');
