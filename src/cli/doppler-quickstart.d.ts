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

/**
 * Extract the generated text from a quickstart result envelope, or
 * throw with an actionable error when the result has no content.
 */
export declare function requireQuickstartContent(
  result: { content?: unknown; modelId?: unknown } | null | undefined
): string;

export declare function main(argv?: string[]): Promise<void>;
