import type { IncomingMessage, ServerResponse } from 'node:http';

export interface ServeSettings {
  port: number;
  host: string;
  model: string | null;
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
  generationOptions: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    topK?: number;
  };
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
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
}

export declare class ServeRequestError extends Error {
  statusCode: number;
  type: string;
  constructor(message: string, statusCode?: number, type?: string);
}

export declare function parseServeArgs(argv: string[]): ServeSettings;

export declare function buildServeReceipt(options: ServeReceiptOptions): {
  schemaVersion: 1;
  runtime: 'doppler-gpu';
  runtimeVersion: string;
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
  generation: {
    maxTokens: number | null;
    temperature: number | null;
    topP: number | null;
    topK: number | null;
  };
  usage: ServeReceiptOptions['usage'];
};

export declare function createServeHandler(dependencies?: ServeDependencies): (
  req: IncomingMessage,
  res: ServerResponse
) => Promise<void>;

export declare function main(argv?: string[]): Promise<void>;
