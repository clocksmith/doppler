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

export function createSequenceProgram(
  hostBridge: { createSequenceProgram(bundle: Record<string, unknown>, options: Record<string, unknown>): unknown },
  bundle: Record<string, unknown>,
  options?: Record<string, unknown>,
): unknown;

export function createRerankProgram(
  hostBridge: { createRerankProgram(bundle: Record<string, unknown>, options: Record<string, unknown>): unknown },
  bundle: Record<string, unknown>,
  options?: Record<string, unknown>,
): unknown;
