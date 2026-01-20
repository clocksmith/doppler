import type { CLIOptions } from '../helpers/types.js';

export { FLAG_SPECS, FLAG_HANDLERS, KNOWN_FLAGS } from './flags.js';
export {
  levenshteinDistance,
  normalizeFlag,
  suggestFlag,
  resolveFlagAlias,
  suggestClosestFlags,
} from './suggestions.js';

export function parseArgs(argv: string[]): CLIOptions;
export function appendRuntimeConfigParams(params: URLSearchParams, opts: CLIOptions): void;
export function setHarnessConfig(opts: CLIOptions, updates: Record<string, unknown>): void;
