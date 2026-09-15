import { createCommandRecorder } from '../../../../gpu/command-recorder.js';
import { recordArgmax, recordGPUSample } from '../../../../gpu/kernels/sample.js';
import { recordHistoryPenalties } from '../../../../gpu/kernels/rep-penalty.js';
import { recordSuppressLogits } from '../../../../gpu/kernels/logit-suppress.js';
import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { GenerationError } from '../../../../config/generation-contract.js';
import { getDevice } from '../../../../gpu/device.js';
import { isDeviceLost } from '../../../../gpu/device-state.js';

export function assertTokenSelectionActive(signal, device) {
  if (signal?.aborted) throw new GenerationError('aborted', 'Generation aborted during GPU token selection.', { cause: signal.reason });
  if (device && (getDevice() !== device || isDeviceLost(device))) {
    throw new Error('GPU token selection device was lost or replaced.');
  }
}

export async function selectTokenFromGpuLogits(result, contextTokens, options, tokenContract) {
  assertTokenSelectionActive(options.signal);
  if (!result?.logitsBuffer || result.logitsDtype !== 'f32'
    || !Number.isSafeInteger(result.vocabSize) || result.vocabSize < 1
    || result.logitsBuffer.size < result.vocabSize * 4) {
    throw new Error('GPU token selection requires a complete, owned f32 logits buffer.');
  }
  const recorder = createCommandRecorder('capsule_token_selection');
  let output, staging;
  let mapped = false;
  try {
    await recordHistoryPenalties(recorder, result.logitsBuffer, contextTokens, {
      ...options, vocabSize: result.vocabSize, logitsDtype: result.logitsDtype,
    });
    await recordSuppressLogits(recorder, result.logitsBuffer, result.vocabSize, options.suppressTokenIds);
    assertTokenSelectionActive(options.signal, recorder.device);
    const sampling = { logitsDtype: result.logitsDtype, outputIndex: 0, logitSoftcap: null,
      padTokenId: tokenContract.padTokenId };
    output = options.temperature === 0
      ? await recordArgmax(recorder, result.logitsBuffer, result.vocabSize, sampling)
      : await recordGPUSample(recorder, result.logitsBuffer, result.vocabSize, {
        ...sampling, temperature: options.temperature, topK: options.topK, topP: options.topP,
        randomSeed: options.seed, greedyThreshold: 0,
      });
    assertTokenSelectionActive(options.signal, recorder.device);
    staging = recorder.device.createBuffer({ label: 'capsule_selected_token', size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    recorder.getEncoder().copyBufferToBuffer(output, 0, staging, 0, 4);
    assertTokenSelectionActive(options.signal, recorder.device);
    recorder.submit();
    await staging.mapAsync(GPUMapMode.READ);
    mapped = true;
    const tokenId = new Uint32Array(staging.getMappedRange())[0];
    assertTokenSelectionActive(options.signal, recorder.device);
    if (tokenId >= result.vocabSize) throw new Error('[Sampling] No finite candidate logits after masking suppressed tokens.');
    return { tokenId, vocabSize: result.vocabSize };
  } finally {
    recorder.abort();
    if (mapped) staging.unmap();
    staging?.destroy();
    if (output) releaseBuffer(output);
  }
}
