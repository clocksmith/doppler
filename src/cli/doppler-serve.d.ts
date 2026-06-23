import type { IncomingMessage, ServerResponse } from 'node:http';

export interface ServeSettings {
  port: number;
  host: string;
  model: string | null;
  modelUrl: string | null;
  help: boolean;
}

export interface ServeRegistryEntry {
  modelId: string;
  sourceCheckpointId: string;
  weightPackId: string;
  manifestVariantId: string;
  artifactCompleteness: string;
  runtimePromotionState: string;
  weightsRefAllowed: boolean;
  aliases: string[];
  modes: string[];
  hf: {
    repoId: string;
    revision: string | null;
    path: string;
  } | null;
}

export interface ServeReceiptOptions {
  requestedModel: string;
  registryEntry: ServeRegistryEntry;
  messages: Array<{
    role: string;
    content: string;
  }>;
  generationOptions: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
  };
  outputContent: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  runtimeModel?: unknown;
}

export interface ServeFailureReceiptOptions {
  requestedModel: string;
  registryEntry: ServeRegistryEntry;
  messages: Array<{
    role: string;
    content: string;
  }>;
  generationOptions: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
  };
  error: unknown;
  runtimeModel?: unknown;
}

export interface ServeDependencies {
  dopplerClient?: {
    chatText(messages: unknown[], options: Record<string, unknown>): Promise<{
      content: string;
      usage: ServeReceiptOptions['usage'];
    }>;
    chat(messages: unknown[], options: Record<string, unknown>): AsyncGenerator<string, void, void>;
  };
  listModels?: () => Promise<ServeRegistryEntry[]>;
  resolveModel?: (model: string) => Promise<ServeRegistryEntry>;
  resolveRuntimeModel?: (registryEntry: ServeRegistryEntry, requestedModel: string) => unknown;
}

export declare class ServeRequestError extends Error {
  statusCode: number;
  type: string;
  constructor(message: string, statusCode?: number, type?: string);
}

export declare function parseServeArgs(argv: string[]): ServeSettings;

export interface ServeReceiptBase {
  receiptVersion: 'doppler_serve_receipt_v1';
  schemaVersion: 1;
  surface: 'serve';
  endpoint: '/v1/chat/completions';
  runtime: 'doppler-gpu';
  runtimeVersion: string;
  runtimePath: 'doppler-gpu.chatText';
  runtimeModelSource:
    | {
      kind: 'quickstart-registry';
      modelId: string;
    }
    | {
      kind: 'url';
      url: string;
    }
    | {
      kind: 'inline-manifest';
      modelId: string;
      baseUrl: string | null;
    };
  modelId: string;
  requestedModel: string;
  resolvedModel: string;
  artifact: {
    format: 'rdrr';
    source: 'quickstart-registry';
    sourceCheckpointId: string;
    weightPackId: string;
    manifestVariantId: string;
    artifactCompleteness: string;
    runtimePromotionState: string;
    weightsRefAllowed: boolean;
    hf: ServeRegistryEntry['hf'];
  };
  request: {
    messages: {
      count: number;
      digest: {
        algorithm: 'sha256';
        value: string;
        bytes: number;
      };
    };
    generationDigest: {
      algorithm: 'sha256';
      value: string;
      bytes: number;
    };
  };
  generation: {
    maxTokens: number | null;
    temperature: number | null;
    topP: number | null;
    topK: number | null;
  };
}

export declare function buildServeReceipt(options: ServeReceiptOptions): ServeReceiptBase & {
  status: 'pass';
  output: {
    role: 'assistant';
    digest: {
      algorithm: 'sha256';
      value: string;
      bytes: number;
    };
    textLength: number;
    empty: boolean;
  };
  transcript: {
    digest: {
      algorithm: 'sha256';
      value: string;
      bytes: number;
    };
  };
  usage: ServeReceiptOptions['usage'];
};

export declare function buildServeFailureReceipt(options: ServeFailureReceiptOptions): ServeReceiptBase & {
  status: 'diagnostic';
  failure: {
    code: string;
    stage: string;
    message: string;
    modelId: string;
    weightLoadFailure: {
      tensorName: string | null;
      tensorRole: string | null;
      tensorDtype: string | null;
      tensorShape: unknown[] | null;
      tensorSizeBytes: number | null;
      tensorLoadStage: string | null;
      toGPU: boolean | null;
      streamedUpload: boolean | null;
      deviceLimitFailure: {
        kind: string | null;
        maxGpuResidentBytes: number | null;
        maxStorageBufferBindingSize: number | null;
        maxBufferSize: number | null;
        maxStorageBuffersPerShaderStage: number | null;
        largeWeightMaxBytes: number | null;
        embeddingKernel: {
          kernel: string | null;
          entry: string | null;
        } | null;
        splitKernelExpected: boolean | null;
        activeSplitKernelMaxSections: number | null;
        maxSplitEmbeddingSections: number | null;
        requiredSplitSections: number | null;
      } | null;
    } | null;
  };
};

export declare function createServeHandler(dependencies?: ServeDependencies): (
  req: IncomingMessage,
  res: ServerResponse
) => Promise<void>;

export declare function main(argv?: string[]): Promise<void>;
