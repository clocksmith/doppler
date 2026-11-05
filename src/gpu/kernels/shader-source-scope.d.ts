export interface ShaderSourceEntry { readonly source: string; readonly digest: string; readonly requiredWgslFeatures: readonly string[]; }
export type ShaderSourceScope = Readonly<object>;
export declare function createShaderSourceScope(sources: Map<string, string>): ShaderSourceScope;
export declare function bindStorageShaderSourceScope(storage: object, scope: ShaderSourceScope): void;
export declare function getStorageShaderSourceScope(storage: object): ShaderSourceScope | null;
export declare function getScopedShaderSource(filename: string): ShaderSourceEntry | null;
export declare function getScopedWgslRequirements(filename: string): readonly string[] | null;
export declare function getShaderScopeCacheKey(): string;
export declare function runWithShaderSourceScope<T>(scope: ShaderSourceScope | null, action: () => T | Promise<T>, signal?: AbortSignal): Promise<T>;
export interface ScopedShaderStream<T> extends AsyncIterator<T, void, unknown>, AsyncIterable<T> {
  /** Settles after iterator cleanup and scope release; rejects on cleanup failure. */
  readonly closed: Promise<void>;
}
export declare function streamWithShaderSourceScope<T>(scope: ShaderSourceScope | null, action: () => AsyncIterable<T>, signal?: AbortSignal): ScopedShaderStream<T>;
