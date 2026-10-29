/** @deprecated Since 0.7.0. Removed in 0.8.0; no replacement export. See CHANGELOG.md. */
import type { FamilyModelEntry } from './family.d.ts';

export declare const FAMILY_ID: string;
export declare const HF_REPO_ID: string | null;
export declare const KNOWN_MODELS: ReadonlyArray<FamilyModelEntry>;
export declare function resolveModel(modelId: string): FamilyModelEntry | null;
export declare function resolveHfBaseUrl(modelId: string, revision?: string): string | null;
