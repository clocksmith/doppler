import type { SourceTruthForgeReceipt } from '../converter/source-truth-forge.js';

/** Read declared local evidence and rerun Forge's fact validation. Never executes reference code. */
export declare function forgeSourceTruthFromFiles(
  spec: Record<string, unknown>,
  sourceRoot: string
): Promise<SourceTruthForgeReceipt>;
