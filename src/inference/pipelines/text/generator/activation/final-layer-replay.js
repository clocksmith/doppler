import { getDevice } from "../../../../../gpu/device.js";
import { createCommandRecorder } from "../../../../../gpu/command-recorder.js";
import { createTensor } from "../../../../../gpu/tensor.js";
import { acquireBuffer, releaseBuffer } from "../../../../../memory/buffer-pool.js";
import { processFFNStandard } from "../../ffn/standard.js";

/** @type {typeof import('./final-layer-replay.js').copyReplayTensor} */
export function copyReplayTensor(tensor, recorder = null) {
  if (tensor.dtype !== 'f32' || tensor.precision !== undefined) throw new Error('Final-layer replay requires ordinary f32 tensors.');
  const device = getDevice();
  if (!device) throw new Error('Final-layer replay requires a GPU device.');
  const bytes = tensor.shape.reduce((a, b) => a * b, 1) * 4;
  const buffer = acquireBuffer(bytes, undefined, 'final_layer_replay_copy');
  try {
    const encoder = recorder?.getEncoder() ?? device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.buffer, 0, buffer, 0, bytes);
    if (!recorder) device.queue.submit([encoder.finish()]);
    return createTensor(buffer, 'f32', [...tensor.shape]);
  } catch (error) { releaseBuffer(buffer); throw error; }
}

/** @type {typeof import('./final-layer-replay.js').replayFinalFFN} */
export async function replayFinalFFN(saved, request) {
  const recorder = createCommandRecorder('final-layer-rotation-replay');
  let output;
  try {
    const input = copyReplayTensor(saved.postAttn, recorder);
    recorder.trackTemporaryBuffer(input.buffer);
    const residual = saved.residual ? copyReplayTensor(saved.residual, recorder) : null;
    if (residual) recorder.trackTemporaryBuffer(residual.buffer);
    const context = { ...saved.context, recorder, decodeBuffers: undefined,
      channelGainRequest: request, debugProbes: undefined, operatorDiagnostics: null,
      trainingCapture: null, debugCheckBuffer: undefined };
    output = await processFFNStandard(saved.layerIdx, input, saved.tokens,
      saved.tokens * context.config.hiddenSize, context, saved.weights, residual, 1, 1, null);
    await recorder.submitAndWait();
    return output;
  } catch (error) {
    recorder.abort();
    if (output) releaseBuffer(output.buffer);
    throw error;
  }
}
