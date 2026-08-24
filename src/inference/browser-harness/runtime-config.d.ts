export declare function resolveRuntime(options: Record<string, unknown>): Record<string, unknown>;
export declare function loadRuntimeConfigFromRef(
  ref: string,
  context: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
export declare function loadRuntimeConfigFromUrl(
  url: string,
  options?: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
export declare function applyRuntimeConfigFromUrl(
  url: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function loadRuntimeProfile(
  profileId: string,
  options?: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
export declare function applyRuntimeProfile(
  profileId: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function applyRuntimeForRun(
  run: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<void>;
