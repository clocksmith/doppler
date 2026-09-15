/** Read requires directives and subgroup identity built-ins, excluding comments. */
export declare function getRequiredWgslFeatures(source: string): string[];
export declare function assertWgslFeaturesSupported(required: readonly string[], available: Iterable<string> | undefined, label: string): void;
