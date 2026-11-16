import type { Tensor } from "../../../../../gpu/tensor.js";
import type { CommandRecorder } from "../../../../../gpu/command-recorder.js";
import type { LayerContext, LayerWeights } from "../../types.js";
import type { ChannelGainRequest } from './channel-gain.js';
export interface FinalLayerReplayCapture {
  layerIdx: number; tokens: number; postAttn: Tensor; residual: Tensor | null;
  context: LayerContext; weights: LayerWeights | undefined;
}
export function copyReplayTensor(tensor: Tensor, recorder?: CommandRecorder | null): Tensor;
/** Internal experimental replay: full-token final FFN only, no attention or cache writes. */
export function replayFinalFFN(saved: FinalLayerReplayCapture, request: ChannelGainRequest): Promise<Tensor>;
