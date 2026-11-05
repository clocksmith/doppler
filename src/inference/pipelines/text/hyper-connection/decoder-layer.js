import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  recordHyperConnectionActivation, recordHyperConnectionBranch, recordHyperConnectionInject, recordHyperConnectionNorm,
  recordHyperConnectionReplicate, runHyperConnectionActivation, runHyperConnectionBranch, runHyperConnectionInject,
  runHyperConnectionNorm, runHyperConnectionReplicate,
} from '../../../../gpu/kernels/hyper-connection.js';

// qwen4_exp decoder-layer composition (Qwen4ExpTextDecoderLayer.forward):
//   streams = streams + PLE(streams)                      (PLE layers only)
//   x, gate = attn_hyper_connection(streams);  streams += gate[s] * attention_or_linear(x)
//   x, gate = mlp_hyper_connection(streams);   streams += gate[s] * moe(x)
// Branches receive the gated-residual branch input directly: no input norm, no
// post-attention norm and no residual of their own. Entry replicates the
// embeddings over the streams; exit is the root mixer without injection and
// replaces the final norm. Branch arithmetic and projections are supplied by
// the caller so their kernels stay declared by the execution plan.

/** @param {unknown} target @returns {target is import('../../../../gpu/command-recorder.js').CommandRecorder} */
function isRecorder(target) {
  return target !== null && typeof (/** @type {any} */ (target))?.beginComputePass === 'function';
}

/** @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} target @param {import('../../../../gpu/tensor.js').Tensor | null} tensor */
function release(target, tensor) {
  if (!tensor) return;
  if (isRecorder(target)) target.trackTemporaryBuffer(tensor.buffer);
  else releaseBuffer(tensor.buffer);
}

/** @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} target */
function operators(target) {
  if (isRecorder(target)) {
    return {
      norm: recordHyperConnectionNorm.bind(null, target),
      activate: recordHyperConnectionActivation.bind(null, target),
      branch: recordHyperConnectionBranch.bind(null, target),
      inject: recordHyperConnectionInject.bind(null, target),
      replicate: recordHyperConnectionReplicate.bind(null, target),
    };
  }
  return {
    norm: runHyperConnectionNorm, activate: runHyperConnectionActivation, branch: runHyperConnectionBranch,
    inject: runHyperConnectionInject, replicate: runHyperConnectionReplicate,
  };
}

/** @type {typeof import('./decoder-layer.js').replicateEmbeddings} */
export function replicateEmbeddings(target, embeds, { hyper, numTokens }) {
  return operators(target).replicate(embeds, { numTokens, streamCount: hyper.streamCount, streamWidth: hyper.streamWidth });
}

/** @type {typeof import('./decoder-layer.js').runGatedResidual} */
export async function runGatedResidual(target, { streams, weights, hyper, numTokens, project, role }) {
  const ops = operators(target);
  const { streamCount, streamWidth, projectionRank, normEps } = hyper;
  const width = streamCount * streamWidth;
  const geometry = { numTokens, streamCount, streamWidth };
  /** @type {import('../../../../gpu/tensor.js').Tensor[]} */
  const temporaries = [];
  const keep = (/** @type {import('../../../../gpu/tensor.js').Tensor} */ tensor) => { temporaries.push(tensor); return tensor; };
  try {
    const normed = keep(await ops.norm(streams, weights.norm, { ...geometry, eps: normEps }));
    const down = keep(await project(`${role}_down`, normed, weights.down, projectionRank, width));
    const lowRank = keep(await ops.activate(down, { mode: 'low_rank_silu', rows: numTokens, columns: projectionRank, streamCount }));
    const up = keep(await project(`${role}_up`, lowRank, weights.up, width, projectionRank));
    const mix = keep(await ops.activate(up, { mode: 'mix_sigmoid', rows: numTokens, columns: width, streamCount }));
    const branchInput = await ops.branch(mix, normed, geometry);
    if (!weights.inject) return { branchInput, gate: null };
    try {
      const logits = keep(await project(`${role}_inject`, normed, weights.inject, streamCount, width));
      const gate = await ops.activate(logits, { mode: 'inject_gate', rows: numTokens, columns: streamCount, streamCount });
      return { branchInput, gate };
    } catch (error) {
      release(target, branchInput);
      throw error;
    }
  } finally {
    for (const tensor of temporaries) release(target, tensor);
  }
}

/**
 * One gated-residual branch: mix, run the branch, inject. Releases the
 * incoming streams when `ownsStreams`.
 * @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} target
 * @param {import('./decoder-layer.js').HyperConnectionBranchStep} step
 */
async function runBranchStep(target, { streams, ownsStreams, weights, hyper, numTokens, project, role, branch }) {
  const { branchInput, gate } = await runGatedResidual(target, { streams, weights, hyper, numTokens, project, role });
  /** @type {import('../../../../gpu/tensor.js').Tensor | null} */
  let output = null;
  try {
    if (!gate) throw new Error(`[HyperConnection] ${role} requires injection weights.`);
    output = await branch(branchInput);
    const injected = await operators(target).inject(streams, output, gate,
      { numTokens, streamCount: hyper.streamCount, streamWidth: hyper.streamWidth });
    if (ownsStreams) release(target, streams);
    return injected;
  } finally {
    release(target, branchInput);
    release(target, gate);
    release(target, output);
  }
}

/** @type {typeof import('./decoder-layer.js').runHyperConnectionDecoderLayer} */
export async function runHyperConnectionDecoderLayer(target, input) {
  const { streams, numTokens, hyper, weights, ple = null, attentionBranch, mlpBranch, project } = input;
  let current = streams;
  let owns = false;
  try {
    if (ple) {
      current = await ple(streams);
      owns = true;
    }
    current = await runBranchStep(target, { streams: current, ownsStreams: owns, weights: weights.attention, hyper, numTokens,
      project, role: 'attn_hc', branch: attentionBranch });
    owns = true;
    return await runBranchStep(target, { streams: current, ownsStreams: true, weights: weights.mlp, hyper, numTokens,
      project, role: 'mlp_hc', branch: mlpBranch });
  } catch (error) {
    if (owns && current !== streams) release(target, current);
    throw error;
  }
}

/** @type {typeof import('./decoder-layer.js').runRootMixer} */
export async function runRootMixer(target, { streams, weights, hyper, numTokens, project }) {
  if (weights.inject) throw new Error('[HyperConnection] the root mixer has no injection weights.');
  const { branchInput } = await runGatedResidual(target, { streams, weights, hyper, numTokens, project, role: 'hc_mixer' });
  return branchInput;
}
