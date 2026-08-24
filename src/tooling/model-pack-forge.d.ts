export declare const FORGE_VERSION: string;
export declare function usage(): string;
export declare function parseArgs(argv: string[]): Record<string, unknown>;
export declare function readJsonInput(value: string): Promise<Record<string, unknown>>;
export declare function buildForgeOptions(
  flags: Record<string, unknown>,
  metaUrl?: string
): Promise<Record<string, unknown>>;
export declare function forgeModelPack(options: Record<string, unknown>): Promise<Record<string, unknown>>;
