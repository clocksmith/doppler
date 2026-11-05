import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { recordHyperConnectionNorm, runHyperConnectionNorm } from '../../../../gpu/kernels/hyper-connection.js';
import { recordPleConv, recordPleGate, recordPleRowDecode, runPleConv, runPleGate, runPleRowDecode } from '../../../../gpu/kernels/ple.js';

// qwen4_exp PLE layer (Qwen4ExpTextPLELayer) on the GPU, added to the residual
// streams before the layer's attention hyper-connection:
//   emb = gathered n-gram rows; key = norm_key(key_proj(emb)); value = value_proj(emb)
//   query = norm_query(streams); gated = sigmoid(signed_sqrt(key . query / sqrt(W))) * value
//   streams' = streams + (gated + silu(dilated_conv(norm_conv(gated))))
// Unpadded single sequences only: the source's padding conv mask is not admitted.

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[PleLayer] ${message}`);
}

/** @param {unknown} target @returns {target is import('../../../../gpu/command-recorder.js').CommandRecorder} */
function isRecorder(target) {
  return target !== null && typeof (/** @type {any} */ (target))?.beginComputePass === 'function';
}

/** Owner of one sequence's PLE short-convolution state ([stateLength, channels] f32, oldest first). */
export class PleConvState {
  /** @param {{ channels: number, stateLength: number }} options */
  constructor({ channels, stateLength }) {
    if (!Number.isSafeInteger(channels) || channels < 1 || !Number.isSafeInteger(stateLength) || stateLength < 1) {
      fail('channels and stateLength must be positive integers.');
    }
    this.channels = channels;
    this.stateLength = stateLength;
    this.epoch = 0;
    this.destroyed = false;
    this.buffer = acquireBuffer(channels * stateLength * 4, undefined, 'ple_conv_state');
    try {
      this.reset();
    } catch (error) {
      releaseBuffer(this.buffer);
      this.destroyed = true;
      throw error;
    }
  }

  /** Zero the state (a new sequence). Advances the epoch. */
  reset() {
    if (this.destroyed) fail('the state has been destroyed.');
    const device = getDevice();
    if (!device) fail('an active GPU device is required to reset the state.');
    device.queue.writeBuffer(this.buffer, 0, new Float32Array(this.channels * this.stateLength));
    this.epoch += 1;
  }

  /** Release after submitted GPU work completes. */
  async destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    try { await getDevice()?.queue.onSubmittedWorkDone(); }
    finally { releaseBuffer(this.buffer); }
  }
}

/** @type {typeof import('./ple-layer.js').runPleLayer} */
export async function runPleLayer(target, { streams, rows, weights, state, ngram, hyper, numTokens, project }) {
  if (!ngram || !hyper) fail('resolved ngramEmbedding and hyperConnections contracts are required.');
  if (!(state instanceof PleConvState) || state.destroyed) fail('a live PleConvState is required.');
  const channels = hyper.streamCount * hyper.streamWidth;
  if (state.channels !== channels || state.stateLength !== ngram.convStateLength) fail('state geometry does not match the contracts.');
  const epoch = state.epoch;
  const recorded = isRecorder(target);
  /** @type {import('../../../../gpu/tensor.js').Tensor[]} */
  const temporaries = [];
  const keep = (/** @type {import('../../../../gpu/tensor.js').Tensor} */ tensor) => {
    temporaries.push(tensor);
    return tensor;
  };
  let retainedGate = null;
  let returned = false;
  const normOptions = { numTokens, streamCount: hyper.streamCount, streamWidth: hyper.streamWidth, eps: ngram.normEps };
  /** @param {import('../../../../gpu/tensor.js').Tensor} input @param {import('../../../../gpu/tensor.js').Tensor} weight */
  const norm = (input, weight) => (recorded
    ? recordHyperConnectionNorm(target, input, weight, normOptions)
    : runHyperConnectionNorm(input, weight, normOptions));
  try {
    const embedding = keep(recorded
      ? await recordPleRowDecode(target, rows, { numTokens, embedDim: ngram.embedDim })
      : await runPleRowDecode(rows, { numTokens, embedDim: ngram.embedDim }));
    const key = keep(await project('ple_key_proj', embedding, weights.keyProj, channels, ngram.embedDim));
    const keyNormed = keep(await norm(key, weights.normKey));
    const value = keep(await project('ple_value_proj', embedding, weights.valueProj, hyper.streamWidth, ngram.embedDim));
    const queryNormed = keep(await norm(streams, weights.normQuery));
    const gateOptions = { numTokens, streamCount: hyper.streamCount, streamWidth: hyper.streamWidth };
    const { gated, gateInput } = recorded
      ? await recordPleGate(target, keyNormed, queryNormed, value, gateOptions)
      : await runPleGate(keyNormed, queryNormed, value, gateOptions);
    retainedGate = gateInput;
    keep(gated);
    const convInput = keep(await norm(gated, weights.normConv));
    if (state.destroyed || state.epoch !== epoch) fail('the convolution state was reset or destroyed during the layer.');
    const convOptions = { numTokens, channels, kernel: ngram.convKernelSize, dilation: ngram.convDilation };
    const output = recorded
      ? await recordPleConv(target, convInput, gated, streams, weights.conv, state.buffer, convOptions)
      : await runPleConv(convInput, gated, streams, weights.conv, state.buffer, convOptions);
    returned = true;
    return { streams: output, gateInput };
  } finally {
    if (!returned && retainedGate) {
      if (recorded) target.trackTemporaryBuffer(retainedGate.buffer);
      else releaseBuffer(retainedGate.buffer);
    }
    for (const tensor of temporaries) {
      if (recorded) target.trackTemporaryBuffer(tensor.buffer);
      else releaseBuffer(tensor.buffer);
    }
  }
}
