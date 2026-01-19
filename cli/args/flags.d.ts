import type { CLIOptions } from '../helpers/types.js';

export interface FlagSpec {
  names: string[];
  handler: (opts: CLIOptions, tokens: string[]) => void;
}

export const FLAG_SPECS: FlagSpec[];
export const FLAG_HANDLERS: Map<string, FlagSpec['handler']>;
export const KNOWN_FLAGS: Set<string>;
