/** Name spelling of a stored format tag. The explicit dtype restores activations omitted when they equal storage. */
export declare function variantNameSuffix(variantTag: string | null | undefined, activationDtype?: string | null): string;

/** Lowercase letters, digits and single hyphens. See docs/style/model-naming.md. */
export declare const MODEL_ID_PATTERN: RegExp;

/** Model IDs issued before the naming rule; each stays valid until its model is reissued under a new identity. */
export declare const LEGACY_NONCANONICAL_MODEL_IDS: readonly string[];

/** Exact IDs that retain compatibility with earlier handwritten or grouped activation suffixes. */
export declare const LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS: readonly string[];

export declare function isCanonicalModelId(modelId: unknown): boolean;

/** Returns the ID, or throws naming the rule, unless the ID is canonical or a listed legacy identity. */
export declare function assertCanonicalModelId(modelId: string, source?: string): string;

/** What is wrong with a name's format suffix for the given manifest variant tag, or null. */
export declare function modelIdSuffixProblem(modelId: unknown, variantTag: string | null | undefined): string | null;

/** True for an identity issued before the naming rule (either list). */
export declare function isLegacyModelId(modelId: unknown): boolean;

/** Returns the ID if it ends with the suffix its format generates or is an earlier identity; throws otherwise. */
export declare function assertModelIdDescribesFormat(modelId: string, variantTag: string | null | undefined, source?: string): string;

/** A conforming label for a model run straight from a file, from the file's name; null when nothing usable remains. */
export declare function modelLabelFromFileName(name: unknown): string | null;

/** Where a manifest's quantizationInfo disagrees with the stored type of its embedding and head tensors. */
export declare function formatRecordProblems(
  quantizationInfo: { weights?: string | null; embeddings?: string | null; lmHead?: string | null } | null | undefined,
  tensors: Record<string, { role?: string; dtype?: string }> | Array<{ role?: string; dtype?: string }> | null | undefined
): string[];
