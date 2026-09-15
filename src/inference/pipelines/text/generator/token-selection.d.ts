import type { ResolvedGenerationOptions } from '../../../../config/generation-contract.js';
export interface GpuLogitsResult { logitsBuffer: GPUBuffer; logitsDtype: 'f32'; vocabSize: number; }
export interface SelectedTokenResult { tokenId: number; vocabSize: number; }
export declare function assertTokenSelectionActive(signal?: AbortSignal | null, device?: GPUDevice): void;
export declare function selectTokenFromGpuLogits(result: GpuLogitsResult, contextTokens: number[],
  options: ResolvedGenerationOptions & { signal?: AbortSignal | null },
  tokenContract: { padTokenId: number | null }): Promise<SelectedTokenResult>;
