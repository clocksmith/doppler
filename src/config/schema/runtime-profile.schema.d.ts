import type { ToolingIntent } from './tooling.schema.js';

export type RuntimeProfileStability = 'canonical' | 'experimental' | 'deprecated';

export interface RuntimeProfileMetadata {
  id: string;
  name: string;
  intent: Exclude<ToolingIntent, null>;
  compatibleIntents: Array<Exclude<ToolingIntent, null>>;
  stability: RuntimeProfileStability;
  owner: string;
  createdAtUtc: string;
}

export function validateRuntimeProfileMetadata(
  profile: Record<string, unknown>,
  label?: string
): Readonly<RuntimeProfileMetadata>;

export function assertRuntimeProfileIntentCompatibility(
  profile: Record<string, unknown>,
  requestIntent: ToolingIntent,
  label?: string
): Readonly<RuntimeProfileMetadata>;
