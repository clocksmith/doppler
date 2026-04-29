import type { ProgramBundle } from '../config/schema/program-bundle.schema.js';

export interface ProgramBundleExportOptions {
  repoRoot?: string;
  manifestPath?: string;
  modelDir?: string;
  referenceReportPath?: string;
  conversionConfigPath?: string | null;
  runtimeConfigPath?: string | null;
  outputPath?: string;
  bundleId?: string;
  createdAtUtc?: string;
  kernelSourceRoot?: string;
  host?: {
    entrypoints?: Array<{
      id: string;
      module: string;
      export: string;
      role: string;
    }>;
    constraints?: Record<string, unknown>;
  };
  captureProfile?: {
    phases?: string[];
    surfaces?: string[];
    adapter?: Record<string, unknown>;
  };
}

export interface ProgramBundleWriteResult {
  outputPath: string;
  bundle: ProgramBundle;
}

export interface ProgramBundleCheckResult {
  ok: true;
  path: string;
  modelId: string;
  bundleId: string;
  artifactCount: number;
  wgslModuleCount: number;
  executionGraphHash: string;
}

export interface ProgramBundleStorageArtifact {
  manifest: Record<string, unknown>;
  modelDir: string;
  manifestPath: string | null;
  manifestRaw: string | null;
}

export declare function resolveProgramBundleStorageArtifact(
  manifest: Record<string, unknown>,
  modelDir: string
): Promise<ProgramBundleStorageArtifact>;

export declare function exportProgramBundle(options?: ProgramBundleExportOptions): Promise<ProgramBundle>;
export declare function writeProgramBundle(options?: ProgramBundleExportOptions): Promise<ProgramBundleWriteResult>;
export declare function loadProgramBundle(bundlePath: string): Promise<ProgramBundle>;
export declare function checkProgramBundleFile(bundlePath: string): Promise<ProgramBundleCheckResult>;
export declare function createProgramBundleCliDefaults(metaUrl: string): { repoRoot: string };
