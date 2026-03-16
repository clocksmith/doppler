export interface HfResolveConfig {
  repoId: string;
  revision?: string | null;
  path: string;
}

export interface HfResolveUrlOptions {
  cdnBasePath?: string;
}

export declare const DEFAULT_HF_CDN_BASE_URL: string;

export declare function buildHfResolveBaseUrl(
  hfConfig: HfResolveConfig | null | undefined,
  options?: HfResolveUrlOptions
): string;
