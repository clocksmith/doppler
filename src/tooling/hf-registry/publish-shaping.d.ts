export declare function isHostedRegistryApprovedEntry(entry: unknown): boolean;
export declare function buildPublishedRegistryEntry(localEntry: unknown, revision: unknown): Record<string, unknown>;
export declare function buildHostedRegistryPayload(
  payload: unknown,
  revisionOverrides?: Map<string, string>
): Record<string, unknown>;
export declare function validateLocalHfEntryShape(entry: unknown): string[];
