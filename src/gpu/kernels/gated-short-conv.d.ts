/**
 * LFM2 gated short convolution kernel.
 *
 * Fuses B*x pre-gating, depthwise causal conv1d, and C*conv_out post-gating
 * into a single GPU dispatch. Each thread handles one channel across all tokens
 * sequentially, maintaining persistent conv state for autoregressive decode.
 */

/** Per-layer state maintained between calls. */
export interface GatedShortConvLayerState {
  /** Pre-dequantized conv1d weights as GPUBuffer, shape [hiddenSize, kernelSize]. */
  convWeightGPU: GPUBuffer;

  /** Persistent conv state as GPUBuffer, shape [hiddenSize, kernelSize - 1]. */
  convStateGPU: GPUBuffer;

  /** Number of channels (hidden dimension). */
  hiddenSize: number;

  /** Conv1d kernel width (e.g., 4). */
  kernelSize: number;
}

/** Tensor returned by the kernel. */
export interface Tensor {
  buffer: GPUBuffer;
  dtype: string;
  shape: readonly number[];
  label: string;
}

/** Options for runGatedShortConvGPU. */
export interface GatedShortConvOptions {
  /** Number of tokens in this batch. Required. */
  numTokens?: number;

  /** Layer index for labeling/tracing. */
  layerIdx?: number;

  /** Command recorder for batched submission. */
  recorder?: {
    getEncoder(): GPUCommandEncoder;
    trackTemporaryBuffer(buffer: GPUBuffer): void;
    beginComputePass(label: string): GPUComputePassEncoder;
    createUniformBuffer(data: ArrayBuffer, label: string): GPUBuffer;
    device: GPUDevice;
  } | null;
}

/**
 * Run the LFM2 gated short convolution on GPU.
 *
 * @param inputTensor Tensor with shape [numTokens, 3 * hiddenSize] containing
 *   concatenated B, C, x from in_proj matmul output.
 * @param layerState Persistent per-layer state (conv weights + conv state buffer).
 * @param options Dispatch options.
 * @returns Output tensor with shape [numTokens, hiddenSize].
 */
export function runGatedShortConvGPU(
  inputTensor: Tensor,
  layerState: GatedShortConvLayerState,
  options?: GatedShortConvOptions
): Promise<Tensor>;
