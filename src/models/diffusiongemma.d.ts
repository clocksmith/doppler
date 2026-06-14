import type { FamilyModelEntry } from './family.d.ts';

export declare const FAMILY_ID: string;
export declare const HF_REPO_ID: string;
export declare const KNOWN_MODELS: ReadonlyArray<FamilyModelEntry>;
export declare function resolveModel(modelId: string): FamilyModelEntry | null;
export declare function resolveHfBaseUrl(modelId: string, revision?: string): string | null;
