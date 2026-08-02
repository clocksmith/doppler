export interface ProgramBundleHostBridge {
  createTextGenerationProgram(
    bundle: Record<string, unknown>,
    options: Record<string, unknown>,
  ): unknown;
}

export declare function createTextGenerationProgram(
  hostBridge: ProgramBundleHostBridge,
  bundle: Record<string, unknown>,
  options?: Record<string, unknown>,
): unknown;
