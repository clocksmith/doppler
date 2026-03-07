import type { ToolingCommandRequest } from '../tooling/command-api.js';

export declare function runTrainingOperatorCommand(
  request: ToolingCommandRequest
): Promise<Record<string, unknown>>;
