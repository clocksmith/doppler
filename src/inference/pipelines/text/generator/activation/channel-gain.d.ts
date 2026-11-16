import type { Tensor } from "../../../../../gpu/tensor.js";
import type { CommandRecorder } from "../../../../../gpu/command-recorder.js";
import type { LayerContext } from "../../types.js";
export interface ChannelGainSpec {
  /** Rotation selects split gate/up + post-SwiGLU rotation on selected dense layers only. */
  transformKind?: 'gain' | 'rotation';
  rotationEpsilon?: number;
  /** Opt-in capture: retain ordinary final-layer FFN inputs for guarded same-sequence replay. */
  retainFinalLayerReplay?: boolean;
  mode: 'capture' | 'apply'; requestId: string; gainHash: string;
  cacheArm: 'base-cache' | 're-prefill'; tokenCount: number;
  span: { start: number; stop: number; templatePositions: number[] };
  layers: { layerIdx: number; channels: number }[];
  /** Historical field name; rotation uses the opaque handle from freezeChannelRotations. */
  frozenGains?: FrozenChannelGains;
  /** Capture-only: original full-prompt content span. Enables same-pass complement reference. */
  captureContentSpan?: { start: number; stop: number };
}
export interface ChannelGainRequest {
  readonly transformKind: 'gain' | 'rotation';
  readonly requestId: string; readonly mode: 'capture' | 'apply'; readonly gainHash: string;
  readonly cacheArm: 'base-cache' | 're-prefill';
  has(layerIdx: number): boolean;
  validateLayer(layerIdx: number, context: LayerContext, tokens: number, channels: number): void;
  activate(layerIdx: number, up: Tensor, gate: Tensor, tokens: number, channels: number,
    swigluLimit: number | null, recorder: CommandRecorder | null): Promise<Tensor>;
  transform(layerIdx: number, tensor: Tensor, tokens: number, channels: number,
    recorder: CommandRecorder | null): Promise<Tensor>;
  captureFinalLayer(layerIdx: number, postAttn: Tensor, residual: Tensor | null, tokens: number,
    context: LayerContext, weights: import("../../types.js").LayerWeights | undefined, supported: boolean): void;
  replayFinalLayer(request: ChannelGainRequest): Promise<Tensor>;
  takeRotations(): Map<number, import("../../../../../gpu/kernels/channel-rotation.js").ChannelRotation>;
  capture(layerIdx: number, tensor: Tensor, tokens: number, channels: number,
    recorder: CommandRecorder | null): Promise<void>;
  /** After submitted prefill completion, transfers energy-buffer ownership to caller. */
  takeEnergies(): Map<number, Tensor>;
  /** Transfers optional same-pass reference buffers independently of instruction buffers. */
  takeReferenceEnergies(): Map<number, Tensor>;
  /** Caller must finish/discard recorders first. Frozen gain buffers are borrowed, never released. */
  close(): void;
}
export interface FrozenChannelGains {
  readonly gainHash: string;
  /** Sum of private GPUBuffer.size at creation, including pool rounding; remains historical after close. */
  readonly gpuBytes: number;
  /** Sum of selected channel counts times four f32 bytes, excluding allocation padding. */
  readonly logicalBytes: number;
  readonly layers: readonly { readonly layerIdx: number; readonly channels: number }[];
  close(): void;
}
/** Copies gains GPU-to-GPU into private buffers, validates positive finite values, and hashes copied bytes.
 * Call only after source-producing recorders settle. Readback is never modified or uploaded. */
export function freezeChannelGains(layers: { layerIdx: number; channels: number; gains: Tensor }[]): Promise<FrozenChannelGains>;
/** Internal research seam; no source weights change. Caller owns frozen gains and their verified hash. */
export function createChannelGainRequest(spec: ChannelGainSpec): ChannelGainRequest;

export function assertChannelGainIdle(state: { isGenerating?: boolean }): void;
export function withChannelGainOperation<T>(state: object, operation: () => Promise<T>): Promise<T>;

/** Copies/validates plane+status and seals angle in opaque immutable request-local ownership. */
export function freezeChannelRotations(layers: { layerIdx: number; channels: number; theta: number; rotation: import("../../../../../gpu/kernels/channel-rotation.js").ChannelRotation }[]): Promise<FrozenChannelGains>;
