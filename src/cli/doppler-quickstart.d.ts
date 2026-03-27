export declare function parseQuickstartArgs(argv: string[]): {
  model: string | null;
  prompt: string | null;
  maxTokens: string | null;
  temperature: string | null;
  json: boolean;
  help: boolean;
  listModels: boolean;
  positionalPrompt: string | null;
};

export declare function readQuickstartConfig(): Promise<{
  schemaVersion: number;
  defaults: {
    model: string;
    prompt: string;
    maxTokens: number;
    temperature: number;
    topK: number;
  };
}>;

export declare function resolveQuickstartSettings(argv?: string[]): Promise<
  | { action: 'help' }
  | { action: 'list-models'; json: boolean }
  | {
    action: 'run';
    json: boolean;
    model: string;
    prompt: string;
    maxTokens: number;
    temperature: number;
    topK: number;
  }
>;

export declare function runQuickstart(settings: {
  json: boolean;
  model: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  topK: number;
}): Promise<{
  ok: true;
  schemaVersion: number;
  version: string;
  requestedModel: string;
  modelId: string;
  prompt: string;
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  elapsedMs: number;
  deviceInfo: Record<string, unknown> | null;
}>;

export declare function main(argv?: string[]): Promise<void>;
