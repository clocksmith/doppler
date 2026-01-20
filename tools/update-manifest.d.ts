/**
 * Update manifest settings without touching shards.
 *
 * Config requirements (tools.updateManifest):
 * - input (string, required)
 * - kernelPath (string|object|null)
 * - clearKernelPath (boolean)
 * - q4kLayout (string|null)
 * - defaultWeightLayout (string|null)
 * - allowUnsafe (boolean)
 * - dryRun (boolean)
 */

export interface UpdateOptions {
  config: string | null;
  help: boolean;
}

export declare function parseArgs(args: string[]): UpdateOptions;
export declare function printHelp(): void;
export declare function resolveManifestPath(input: string): Promise<string>;
export declare function main(): Promise<void>;
