export declare function asOptionalString(value: unknown, label: string): string | null;
export declare function asOptionalBoolean(value: unknown, label: string): boolean | null;
export declare function asOptionalObject(value: unknown, label: string): Record<string, unknown> | null;
export declare function asOptionalStringArray(value: unknown, label: string): string[] | null;
export declare function asOptionalPositiveInteger(value: unknown, label: string): number | null;
export declare function asOptionalTrainingStage(value: unknown, label: string): string | null;
export declare function asOptionalForceResumeReason(value: unknown, label: string): string | null;
export declare function asOptionalAction(value: unknown, label: string, allowed: string[]): string | null;
export declare function assertCommand(value: unknown): string;
export declare function resolveCommandRuntimeContract(command: string): { suite: string | null; intent: string | null };
export declare function asOptionalCacheMode(value: unknown, label: string): 'cold' | 'warm' | null;
export declare function asOptionalLoadMode(value: unknown, label: string): 'opfs' | 'http' | 'memory' | null;
export declare function assertModelId(value: unknown, command: string, suite: string): string;
export declare function assertForbiddenStringField(raw: Record<string, unknown>, fieldName: string, command: string): void;
export declare function assertForbiddenObjectField(raw: Record<string, unknown>, fieldName: string, command: string): void;
export declare function assertForbiddenConfigChainField(raw: Record<string, unknown>, command: string): void;
export declare function resolveSuiteForCommand(
  raw: Record<string, unknown>,
  command: string,
  runtimeContract: { suite: string | null; intent: string | null }
): string;
export declare function createCommandRequestBase(
  raw: Record<string, unknown>,
  command: string
): Record<string, unknown>;
