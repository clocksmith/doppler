import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { ResolvedHyperConnections } from '../../../../config/schema/hyper-connection-decoder.schema.js';

/** Caller-owned projection; roles are `${attn_hc|mlp_hc|hc_mixer}_{down|up|inject}`. */
export type HyperConnectionProjection = (
  role: string,
  input: Tensor,
  weight: unknown,
  outFeatures: number,
  inFeatures: number
) => Promise<Tensor>;

export interface GatedResidualWeights {
  /** hc_norm weight [streamCount * streamWidth] f32. */
  norm: Tensor;
  down: unknown;
  up: unknown;
  /** block_inject_weight; null for the root mixer. */
  inject: unknown | null;
}

export interface GatedResidualInput {
  streams: Tensor;
  weights: GatedResidualWeights;
  hyper: ResolvedHyperConnections;
  numTokens: number;
  project: HyperConnectionProjection;
  role: string;
}

export interface HyperConnectionBranchStep extends Omit<GatedResidualInput, 'role'> {
  role: string;
  ownsStreams: boolean;
  branch: (branchInput: Tensor) => Promise<Tensor>;
}

export interface HyperConnectionDecoderLayerInput {
  /** Caller-owned streams [numTokens, streamCount * streamWidth]; not released. */
  streams: Tensor;
  numTokens: number;
  hyper: ResolvedHyperConnections;
  weights: { attention: GatedResidualWeights; mlp: GatedResidualWeights };
  /** PLE on the incoming streams for PLE layers; returns new streams owned by the layer. */
  ple?: ((streams: Tensor) => Promise<Tensor>) | null;
  /** Attention or linear attention on the branch input [numTokens, streamWidth]. */
  attentionBranch: (branchInput: Tensor) => Promise<Tensor>;
  /** MoE on the branch input [numTokens, streamWidth]. */
  mlpBranch: (branchInput: Tensor) => Promise<Tensor>;
  project: HyperConnectionProjection;
}

export declare function replicateEmbeddings(
  target: CommandRecorder | null,
  embeds: Tensor,
  options: { hyper: ResolvedHyperConnections; numTokens: number }
): Promise<Tensor>;

/** Branch input [numTokens, streamWidth] and injection gate [numTokens, streamCount] (null without inject weights). */
export declare function runGatedResidual(
  target: CommandRecorder | null,
  input: GatedResidualInput
): Promise<{ branchInput: Tensor; gate: Tensor | null }>;

/** New streams owned by the caller. */
export declare function runHyperConnectionDecoderLayer(
  target: CommandRecorder | null,
  input: HyperConnectionDecoderLayerInput
): Promise<Tensor>;

/** Root mixer output [numTokens, streamWidth]; replaces the final norm before the LM head. */
export declare function runRootMixer(
  target: CommandRecorder | null,
  input: Omit<GatedResidualInput, 'role'>
): Promise<Tensor>;
