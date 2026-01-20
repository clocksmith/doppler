export interface VfsManifestFile {
  path: string;
  url?: string;
  contentType?: string;
  size?: number;
}

export interface VfsManifest {
  version?: number;
  generatedAt?: string;
  root?: string;
  files: VfsManifestFile[];
}

export interface SeedProgress {
  path: string;
  index: number;
  total: number;
  status: 'seeded' | 'skipped';
}

export interface SeedOptions {
  preserve?: boolean;
  timeoutMs?: number;
  onProgress?: (progress: SeedProgress) => void;
}

export interface SeedResult {
  total: number;
  seeded: number;
  skipped: number;
}

export declare function loadVfsManifest(manifestUrl: string): Promise<VfsManifest>;
export declare function seedVfsFromManifest(
  manifest: VfsManifest,
  options?: SeedOptions
): Promise<SeedResult>;
