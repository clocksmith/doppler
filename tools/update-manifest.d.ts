/**
 * Update manifest settings without touching shards.
 *
 * Safe edits (default):
 * - optimizations.kernelPath
 *
 * Unsafe edits (require --allow-unsafe):
 * - config.q4kLayout
 * - defaultWeightLayout
 */

export interface UpdateOptions {
  input: string | null;
  kernelPath: string | Record<string, unknown> | null;
  clearKernelPath: boolean;
  q4kLayout: string | null;
  defaultWeightLayout: string | null;
  allowUnsafe: boolean;
  dryRun: boolean;
  help: boolean;
}

export declare function parseArgs(args: string[]): UpdateOptions;
export declare function printHelp(): void;
export declare function resolveManifestPath(input: string): Promise<string>;
export declare function main(): Promise<void>;
