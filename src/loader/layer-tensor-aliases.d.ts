export declare const LAYER_PREFIXES: (layerIndex: number) => string[];
export declare const ATTN_SUFFIXES: Readonly<Record<string, readonly string[]>>;
export declare const LINEAR_ATTN_SUFFIXES: Readonly<Record<string, readonly string[]>>;
export declare const CONV_SUFFIXES: Readonly<Record<string, readonly string[]>>;
export declare const FFN_SUFFIXES: Readonly<Record<string, readonly string[]>>;
export declare const ROUTER_SUFFIXES: Readonly<Record<string, readonly string[]>>;
export declare const SINK_SUFFIXES: readonly string[];
export declare const MATMUL_KEYS: readonly string[];
export declare const LAYER_WEIGHT_CONSUMERS: readonly Readonly<{ role: string; suffixes: readonly string[]; additionalRoles: readonly string[] }>[];
