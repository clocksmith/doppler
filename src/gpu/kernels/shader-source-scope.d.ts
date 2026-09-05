export type ShaderSourceScope = Readonly<object>;
export declare function createShaderSourceScope(sources: Map<string, string>): ShaderSourceScope;
export declare function bindStorageShaderSourceScope(storage: object, scope: ShaderSourceScope): void;
export declare function getStorageShaderSourceScope(storage: object): ShaderSourceScope | null;
export declare function getScopedShaderSource(filename: string): { source: string; digest: string } | null;
export declare function getShaderScopeCacheKey(): string;
export declare function runWithShaderSourceScope<T>(scope: ShaderSourceScope | null, action: () => T | Promise<T>): Promise<T>;
export declare function streamWithShaderSourceScope<T>(scope: ShaderSourceScope | null, action: () => AsyncIterable<T>): AsyncGenerator<T>;
