export interface WrappedPipelineHandle {
  readonly loaded: boolean;
  readonly modelId: string;
  readonly manifest: Record<string, unknown> | null;
  readonly deviceInfo: Record<string, unknown> | null;
  generateText(prompt: unknown, opts?: Record<string, unknown>): Promise<string>;
  unload(): Promise<void>;
}

export declare function wrapPipelineAsHandle(
  pipeline: {
    generate: (...args: unknown[]) => AsyncIterable<unknown>;
    [key: string]: unknown;
  },
  resolved?: {
    modelId?: string;
    manifest?: Record<string, unknown>;
    deviceInfo?: Record<string, unknown>;
  }
): WrappedPipelineHandle;
