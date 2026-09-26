export declare const RIG_VERSION: string;
export declare function usage(): string;
export declare function parseArgs(argv: string[]): Record<string, unknown>;
export declare function readJsonInput(value: string): Promise<Record<string, unknown>>;
export declare function buildRigOptions(
  flags: Record<string, unknown>,
  metaUrl?: string
): Promise<Record<string, unknown>>;
export declare function rigModelCapsule(options: Record<string, unknown>): Promise<Record<string, unknown>>;
