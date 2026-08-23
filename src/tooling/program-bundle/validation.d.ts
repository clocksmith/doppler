export declare function normalizeDigest(value: unknown, label: string): string;
export declare function requirePlainObject(value: unknown, label: string): Record<string, unknown>;
export declare function requireString(value: unknown, label: string): string;
export declare function assertSha256ShardHashAlgorithm(
  hashAlgorithm: unknown,
  filename: string
): void;
