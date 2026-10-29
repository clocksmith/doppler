export interface FamilyModelEntry {
  readonly modelId: string;
  readonly label: string;
  readonly sourceModel: string;
  readonly hfRepoId: string | null;
  readonly hfPath: string | null;
  readonly modes: ReadonlyArray<string>;
}

export interface FamilyModule {
  readonly FAMILY_ID: string;
  /** The family's Hugging Face repository when all hosted entries share one; otherwise null. */
  readonly HF_REPO_ID: string | null;
  readonly KNOWN_MODELS: ReadonlyArray<FamilyModelEntry>;
  resolveModel(modelId: string): FamilyModelEntry | null;
  resolveHfBaseUrl(modelId: string, revision?: string): string | null;
}

export interface CreateFamilyInput {
  /** Catalog `family` value; entries derive from models/catalog.json. */
  readonly familyId: string;
}

export declare function createFamily(input: CreateFamilyInput): FamilyModule;
