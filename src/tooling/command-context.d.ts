import type {
  ToolingCommand,
  ToolingCommandRequest,
  ToolingIntent,
  ToolingWorkload,
} from './command-api.js';

export const COMMAND_CONTEXT_SCHEMA_VERSION: 1;

export interface CommandContext {
  readonly schemaVersion: typeof COMMAND_CONTEXT_SCHEMA_VERSION;
  readonly command: ToolingCommand;
  readonly workload: ToolingWorkload | null;
  readonly intent: ToolingIntent;
}

export function createCommandContext(request: ToolingCommandRequest): CommandContext;
export function assertCommandContextMatchesOptions(
  commandContext: CommandContext,
  options?: {
    command?: ToolingCommand;
    workload?: ToolingWorkload | null;
    intent?: ToolingIntent;
  }
): CommandContext;
