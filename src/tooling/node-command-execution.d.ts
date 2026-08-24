import type { ToolingCommandRequestInput } from './command-api.js';
import type {
  NodeCommandRunOptions,
  NodeCommandRunResult,
} from './node-command-runner.js';

export declare function hasNodeWebGPUSupport(): boolean;

export declare function runNodeCommandExecution(
  commandRequest: ToolingCommandRequestInput,
  options?: NodeCommandRunOptions
): Promise<NodeCommandRunResult>;
