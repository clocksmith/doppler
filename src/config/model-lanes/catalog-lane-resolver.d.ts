export interface CatalogModelLane {
  modelId?: string;
  label?: string;
  modes?: string[];
  quickstart?: boolean;
  demoVisible?: boolean;
  demoPreferredVariantId?: string;
  artifactCompleteness?: 'complete' | 'weights-ref' | 'incomplete' | string;
  runtimePromotionState?: string;
  weightsRefAllowed?: boolean;
  weightPackId?: string;
  localBaseUrl?: string | null;
  baseUrl?: string | null;
  hf?: {
    repoId?: string;
    revision?: string;
    path?: string;
  } | null;
  lifecycle?: {
    status?: {
      runtime?: string;
      tested?: string;
    };
  };
  demoFallbackVariant?: CatalogModelLane | null;
  weightsRefPrimary?: string | null;
  laneSelection?: {
    kind: 'primary' | 'preferred_weights_ref';
    visibleModelId: string;
    preferredModelId: string;
    fallbackModelId: string | null;
    weightPackId: string | null;
  };
  [key: string]: unknown;
}

export declare function isWeightsRefLane(entry: CatalogModelLane | null | undefined): boolean;

export declare function isPrimaryWeightPackLane(
  entry: CatalogModelLane | null | undefined,
  weightPackId: string | null | undefined
): boolean;

export declare function isManifestOwnedLane(entry: CatalogModelLane | null | undefined): boolean;

export declare function isVerifiedManifestOwnedLane(entry: CatalogModelLane | null | undefined): boolean;

export declare function findPrimaryForWeightPack(
  catalogEntries: CatalogModelLane[] | null | undefined,
  weightPackId: string | null | undefined
): CatalogModelLane | null;

export declare function findRegisteredSiblingsOf(
  primaryEntry: CatalogModelLane | null | undefined,
  catalogEntries: CatalogModelLane[] | null | undefined,
  storedModelIds: Set<string> | string[] | null | undefined
): CatalogModelLane[];

export declare function selectCatalogModelLanes(
  catalogEntries: CatalogModelLane[] | null | undefined,
  options?: {
    localBaseUrls?: Map<string, string>;
    isVisibleEntry?: (entry: CatalogModelLane) => boolean;
    hasSource?: (entry: CatalogModelLane) => boolean;
  }
): CatalogModelLane[];
