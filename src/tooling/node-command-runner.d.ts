import type { RuntimeConfigLoadOptions } from '../inference/browser-harness.js';
import type {
  ToolingCommandRequest,
  ToolingCommandRequestInput,
} from './command-api.js';

export interface NodeCommandRunOptions {
  runtimeLoadOptions?: RuntimeConfigLoadOptions;
  onProgress?: (progress: {
    stage: string | null;
    current: number | null;
    total: number | null;
    message: string | null;
  }) => void;
}

export interface NodeCommandRunResult {
  ok: true;
  surface: 'node';
  request: ToolingCommandRequest;
  result: unknown;
}

export declare function hasNodeWebGPUSupport(): boolean;

export declare function runNodeCommand(
  commandRequest: ToolingCommandRequestInput,
  options?: NodeCommandRunOptions
): Promise<NodeCommandRunResult>;

export declare function normalizeNodeCommand(
  commandRequest: ToolingCommandRequestInput
): ToolingCommandRequest;
