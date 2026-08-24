export interface WgslClosure {
  modules: Array<Record<string, unknown> & { id: string; digest: string }>;
  packageFiles: Array<Record<string, unknown> & {
    path: string;
    contents: string;
    artifact: Record<string, unknown>;
  }>;
  kernelClosure: Record<string, unknown>;
}

export function buildExecutionStepMetadata(
  execution: Record<string, unknown>,
  expandedSteps: Array<Record<string, unknown>>,
  modules: WgslClosure['modules']
): { steps: Array<Record<string, unknown>>; stepMetadataHash: string };
export function buildWgslClosure(
  execution: Record<string, unknown>,
  expandedSteps: Array<Record<string, unknown>>,
  options: { repoRoot: string; kernelSourceRoot?: string }
): Promise<WgslClosure>;
