export interface FamilyModelEntry {
  readonly modelId: string;
  readonly label: string;
  readonly sourceModel: string;
  readonly hfPath: string;
  readonly defaultRuntimeProfile: string;
  readonly modes: ReadonlyArray<'text' | 'vision' | 'embedding' | 'translate' | 'diffusion'>;
}

export interface FamilyModule {
  readonly FAMILY_ID: string;
  readonly HF_REPO_ID: string;
  readonly KNOWN_MODELS: ReadonlyArray<FamilyModelEntry>;
  resolveModel(modelId: string): FamilyModelEntry | null;
  resolveHfBaseUrl(modelId: string, revision?: string): string | null;
}

export interface CreateFamilyInput {
  readonly familyId: string;
  readonly hfRepoId: string;
  readonly knownModels: ReadonlyArray<FamilyModelEntry>;
}

export declare function createFamily(input: CreateFamilyInput): FamilyModule;
