import type { ToolingIntent } from './command-api.js';
import type { RuntimeInputDocument } from './runtime-input-composition.js';

export function getLegacyRuntimeIntent(runtimeConfig: Record<string, unknown> | null): ToolingIntent;
export function stripLegacyRuntimeIntent<T extends Record<string, unknown> | null>(runtimeConfig: T): T;
export function assertRuntimeInputIntentCompatibility(
  requestIntent: ToolingIntent,
  documents: ReadonlyArray<RuntimeInputDocument>
): true;
