export declare const PROGRAM_BUNDLE_SCHEMA_VERSION: 1;
export declare const PROGRAM_BUNDLE_SCHEMA_ID: 'doppler.program-bundle/v1';
export declare const PROGRAM_BUNDLE_HOST_SCHEMA_ID: 'doppler.host-js/v1';
export declare const PROGRAM_BUNDLE_HOST_JS_SUBSET: 'doppler-webgpu-host/v1';
export declare const PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID: 'doppler.capture-profile/v1';
export declare const PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID: 'doppler.reference-transcript/v1';

export interface ProgramBundleArtifact {
  role: 'manifest' | 'weight-shard' | 'tokenizer' | 'conversion-config' | 'runtime-config' | 'reference-report' | 'source' | 'other';
  path: string;
  hash: string;
  sizeBytes: number | null;
}

export interface ProgramBundleWgslModule {
  id: string;
  file: string;
  entry: string;
  digest: string;
  sourcePath: string | null;
  reachable: boolean;
  metadata: {
    entry: string;
    sourceMetadataHash: string;
    bindings: Array<{
      group: number;
      binding: number;
      addressSpace: string | null;
      access: string | null;
      name: string;
    }>;
    overrides: Array<{
      name: string;
      type: string | null;
      defaultValue: string | null;
    }>;
    workgroupSize: string[];
    requiresSubgroups: boolean;
  };
}

export interface ProgramBundleReferenceTranscript {
  schema: typeof PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID;
  source: {
    kind: string;
    path: string;
    hash: string;
  };
  executionGraphHash: string;
  surface: string | null;
  prompt: {
    identity: string;
    hash: string;
    tokenIdsHash: string | null;
    tokenCount: number | null;
  };
  output: {
    textHash: string;
    tokensGenerated: number;
    stopReason: string;
    stopTokenId: number | null;
  };
  tokens: {
    ids?: number[];
    generatedTokenIdsHash: string;
    generatedTextHash: string;
    preview: Array<Record<string, unknown>>;
    perStep: Array<{
      index: number;
      tokenId: number;
      tokenHash: string;
    }>;
    coverage?: {
      mode: string;
      omitted: number;
    };
  };
  phase: {
    prefillMs: number | null;
    decodeMs: number | null;
    prefillTokens: number | null;
    decodeTokens: number | null;
  };
  kvCache: {
    mode: string;
    layout: string | null;
    kvDtype: string | null;
    seqLen: number | null;
    maxSeqLen: number | null;
    usedBytes: number | null;
    allocatedBytes: number | null;
    counters: Record<string, unknown> | null;
    stateHash: string;
    byteDigestMode?: string | null;
    byteDigest?: string | null;
    byteDigests?: Array<{
      layer: number;
      seqLen: number;
      keyBytes: number;
      valueBytes: number;
      keyDigest: string;
      valueDigest: string;
    }> | null;
  };
  logits: {
    mode: 'not-captured' | 'sha256-per-step';
    reason?: string | null;
    perStepDigests?: string[] | null;
    steps?: Array<{
      index: number | null;
      tokenId: number | null;
      inputTokenCount: number | null;
      dtype: string;
      elementCount: number;
      digest: string;
    }>;
  };
  tolerance: {
    tokenPolicy: string;
    logitsPolicy: string;
  };
}

export interface ProgramBundle {
  schema: typeof PROGRAM_BUNDLE_SCHEMA_ID;
  schemaVersion: typeof PROGRAM_BUNDLE_SCHEMA_VERSION;
  bundleId: string;
  modelId: string;
  createdAtUtc: string;
  sources: {
    manifest: {
      path: string;
      hash: string;
    };
    conversionConfig: {
      path: string;
      hash: string;
    } | null;
    executionGraph: {
      schema: string | null;
      hash: string;
      expandedStepHash: string;
    };
    weightSetHash: string;
    artifactSetHash: string;
  };
  host: {
    schema: typeof PROGRAM_BUNDLE_HOST_SCHEMA_ID;
    jsSubset: typeof PROGRAM_BUNDLE_HOST_JS_SUBSET;
    entrypoints: Array<{
      id: string;
      module: string;
      export: string;
      role: string;
      sourceHash?: string;
      validation?: {
        dynamicImport: 'none-detected';
        dom: 'none-detected';
      };
    }>;
    constraints: {
      dynamicImport: 'disallowed';
      dom: 'disallowed-in-model-path';
      filesystem: 'declared-artifacts-only';
      network: 'declared-artifacts-only';
    };
  };
  wgslModules: ProgramBundleWgslModule[];
  execution: {
    graphHash: string;
    stepMetadataHash: string;
    kernelClosure: {
      declaredKernelIds: string[];
      reachableKernelIds: string[];
      excludedKernelIds: string[];
      undeclaredKernelRefs: string[];
      expandedStepCount: number;
      phases: {
        prefill: number;
        decode: number;
        preLayer: number;
        postLayer: number;
      };
    };
    steps: Array<{
      id: string;
      index: number;
      op: string;
      phase: string;
      section: string;
      layers: 'all' | number[];
      src: string;
      dst: string;
      kernelId: string;
      kernel: string;
      entry: string;
      kernelDigest: string;
      weights: string | null;
      constants: Record<string, unknown> | null;
      precision: Record<string, unknown> | null;
      dispatch: {
        phase: string;
        workgroups: string;
        bindings: Array<{
          group: number;
          binding: number;
          name: string;
          addressSpace: string | null;
          access: string | null;
        }>;
      };
    }>;
  };
  captureProfile: {
    schema: typeof PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID;
    deterministic: true;
    phases: string[];
    surfaces: string[];
    adapter: Record<string, unknown> & {
      source: string;
    };
    hashPolicy: {
      graph: string;
      dispatch: string;
      transcript: string;
    };
    captureHash: string;
  };
  artifacts: ProgramBundleArtifact[];
  referenceTranscript: ProgramBundleReferenceTranscript;
}

export declare function validateProgramBundle(bundle: unknown): ProgramBundle;
