#!/usr/bin/env node
/**
 * Purge a model from OPFS cache (browser storage).
 * Requires a browser context because OPFS is origin-scoped.
 */

export interface PurgeOptions {
  model: string | null;
  baseUrl: string;
  headless: boolean;
  noServer: boolean;
  profileDir: string | null;
  verbose: boolean;
  help: boolean;
}

export declare function parseArgs(argv: string[]): PurgeOptions;
export declare function printHelp(): void;
export declare function main(): Promise<void>;
