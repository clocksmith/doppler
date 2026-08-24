export declare function canonicalizeJson(value: unknown): string;
export declare function hashBytesSha256(bytes: Uint8Array | ArrayBuffer | ArrayBufferView): string;
export declare function computeCanonicalSha256(value: unknown): string;
export declare function computeNamespacedCanonicalSha256(
  namespace: 'artifact' | 'transcript' | 'plan' | 'integrity',
  value: unknown
): string;
