export interface ProgramBundleArtifact {
  role: string;
  path: string;
  hash: string;
  sizeBytes: number;
}

export interface PackageSourceFile extends ProgramBundleArtifact {
  contents: string;
  artifact: ProgramBundleArtifact;
}

export function hashStableJson(value: unknown): string;
export function normalizeSlash(value: unknown): string;

export function createPackageSourceFile(options: {
  role: string;
  id: string;
  extension: string;
  source: unknown;
}): PackageSourceFile;

export function toRepoRelativePath(filePath: string, repoRoot: string): string;
export function readTextFile(filePath: string, label: string): Promise<string>;
export function tryReadTextFile(filePath: string): Promise<string | null>;
export function readJsonFile<T = unknown>(
  filePath: string,
  label: string
): Promise<{ raw: string; json: T }>;

export function fileArtifact(options: {
  role: string;
  filePath: string;
  repoRoot: string;
  artifactPath?: string | null;
}): Promise<ProgramBundleArtifact>;

export function buildReferenceTranscript(
  referenceReportPath: string,
  repoRoot: string,
  executionGraphHash: string
): Promise<{
  artifact: ProgramBundleArtifact;
  adapter: Record<string, unknown>;
  transcript: Record<string, unknown>;
}>;
