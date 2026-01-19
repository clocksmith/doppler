import type { CLIOptions, SuiteType } from '../helpers/types.js';

export { FLAG_SPECS, FLAG_HANDLERS, KNOWN_FLAGS } from './flags.js';
export {
  levenshteinDistance,
  normalizeFlag,
  suggestFlag,
  resolveFlagAlias,
  suggestClosestFlags,
} from './suggestions.js';

export function normalizeSuite(suite: string): SuiteType | string;
export function parseArgs(argv: string[]): CLIOptions;
export function hasCliFlag(opts: CLIOptions, flags: string[]): boolean;
export function appendRuntimeConfigParams(params: URLSearchParams, opts: CLIOptions): void;
export function setHarnessConfig(opts: CLIOptions, updates: Record<string, unknown>): void;
