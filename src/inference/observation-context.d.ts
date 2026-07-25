import type { RuntimeConfigSchema } from '../config/schema/index.js';
import type { CommandContext } from '../tooling/command-context.js';

export const OBSERVATION_CONTEXT_SCHEMA: 'doppler.observation-context/v1';

export interface ObservationContext {
  readonly schema: typeof OBSERVATION_CONTEXT_SCHEMA;
  readonly commandContext: CommandContext | null;
  readonly diagnostics: 'off' | 'on_failure' | 'always';
  readonly probes: ReadonlyArray<Readonly<Record<string, unknown>>>;
  readonly tracing: {
    readonly pipelineEnabled: boolean;
    readonly layers: ReadonlyArray<number> | null;
    readonly kernelTrace: Readonly<Record<string, unknown>> | null;
  };
  readonly receiptPolicy: 'off' | 'on_failure' | 'always';
}

export function createObservationContext(options: {
  runtimeConfig: RuntimeConfigSchema;
  commandContext?: CommandContext | null;
}): ObservationContext;
