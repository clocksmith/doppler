import type { Tensor } from '../../../../gpu/tensor.js';

export interface PrefixEmbeddingOverride {
  prefixLength: number;
  offset: number;
  embeddings: Float32Array | GPUBuffer;
  expectedLength: number;
  byteLength: number;
  byteOffset: number;
}

export interface InputSpan {
  offset: number;
  length: number;
}

export type PrefixEmbeddingOverrideTransitionDeclaration =
  | 'step_precision'
  | 'explicit_cast_step'
  | null;

export interface PrefixEmbeddingOverrideExecutionOptions {
  executionPolicies?: Record<string, unknown> | null;
  transitionDeclaredBy?: PrefixEmbeddingOverrideTransitionDeclaration;
}

export declare function normalizePrefixEmbeddingOverride(
  override: Record<string, unknown> | null | undefined,
  hiddenSize: number,
  numTokens: number,
  contextLabel: string
): PrefixEmbeddingOverride | null;

export declare function resolvePrefillEmbeddingInputIds(
  inputIds: readonly number[],
  embeddingInputSpan: InputSpan & { tokenId: number } | null | undefined,
  contextLabel: string
): number[] | readonly number[];

export declare function resolvePrefillMultimodalBidirectionalSpan(
  inputIds: readonly number[],
  bidirectionalSpan: InputSpan | null | undefined,
  contextLabel: string
): InputSpan | null;

export declare function resolvePrefixEmbeddingOverrideTransitionDeclaredBy(
  executionV1State: Record<string, unknown> | null | undefined
): PrefixEmbeddingOverrideTransitionDeclaration;

export declare function applyPrefixEmbeddingOverride(
  baseTensor: Tensor,
  override: PrefixEmbeddingOverride | null,
  hiddenSize: number,
  contextLabel: string,
  executionOptions?: Record<string, unknown> | PrefixEmbeddingOverrideExecutionOptions | null
): Promise<Tensor>;
