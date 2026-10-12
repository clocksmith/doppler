import { GENERATION_CONTRACT } from '../../../config/generation-contract.js';
import { contractAssert as assert, exactRecord, admitTokens, count } from '../../../config/canonical-contract-codec.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { clonePartitionAttempt } from './partition-attempt.js';
import { executePartitionLayers } from './partition-execution.js';
import { resolveActivationTensorLayout } from './layer-partition-contract.js';

/** @type {import('./partition-replay.js').replayPartitionAttempt} */
export async function replayPartitionAttempt(parent, inputs, signal) {
  signal.throwIfAborted();
  const partition = parent.state.modelPartition;
  assert(partition && (partition.index === 0 || partition.index === 1), 'Replay requires a fixed two-stage partition.');
  const plan = partition.plan;
  const layout = resolveActivationTensorLayout({ dtype: plan.activationDtype,
    precision: plan.activationPrecision, components: plan.activationComponents,
    elementStrideBytes: plan.activationElementStrideBytes });
  const rowBytes = count(plan.hiddenSize * layout.elementStrideBytes, 'replay row bytes');
  assert(rowBytes > 0, 'Replay requires nonempty boundary rows.');
  const maxRows = GENERATION_CONTRACT.canonical.maxProposalTokens + 1;
  const cache = parent.state.kvCache;
  assert(cache, 'Replay requires a live parent cache.');
  /** @type {readonly number[] | null} */
  let tokens = null;
  /** @type {readonly Uint8Array[]} */
  let retained;
  if (partition.index === 0) {
    const source = /** @type {Extract<import('./partition-replay.js').PartitionReplayInputs, { tokenIds: readonly number[] }>} */
      (exactRecord(inputs, ['tokenIds', 'boundaryRows']));
    tokens = admitTokens(source.tokenIds, plan.vocabSize, maxRows);
    retained = source.boundaryRows;
    assert(Array.isArray(retained) && retained.length === tokens.length, 'Replay tokens and boundary row counts differ.');
  } else {
    retained = /** @type {Extract<import('./partition-replay.js').PartitionReplayInputs, { activationRows: readonly Uint8Array[] }>} */
      (exactRecord(inputs, ['activationRows'])).activationRows;
  }
  assert(Array.isArray(retained) && retained.length > 0 && retained.length <= maxRows
    && retained.length <= cache.maxSeqLen - parent.state.currentSeqLen, 'Replay exceeds accepted row capacity.');
  for (const row of retained) assert(row instanceof Uint8Array && row.byteLength === rowBytes, 'Replay boundary row byte length mismatch.');
  const rows = retained.map(row => row.slice());
  const child = await clonePartitionAttempt(parent, signal);
  try {
    for (let i = 0; i < rows.length; i++) {
      signal.throwIfAborted();
      const result = await executePartitionLayers(child.state, { numTokens: 1,
        ...(tokens ? { tokenIds: [tokens[i]] } : { activationBytes: rows[i] }) }, signal);
      try {
        if (tokens) {
          const actual = result.activationBytes && new Uint8Array(result.activationBytes);
          if (!actual || actual.length !== rows[i].length || !actual.every((byte, index) => byte === rows[i][index])) {
            throw Object.assign(new Error('Accepted boundary replay differs from the retained row.'), { code: 'REPLAY_DIVERGENCE' });
          }
        } else assert(result.logits, 'Output-stage replay did not execute its output layers.');
      } finally { if (result.logits) releaseBuffer(result.logits.logitsBuffer); }
      child.state.currentSeqLen++;
      child.state.decodeStepCount++;
    }
    await child.state.gpuContext?.device?.queue.onSubmittedWorkDone();
    signal.throwIfAborted();
    return child;
  } catch (error) {
    try { await child.state.gpuContext?.device?.queue.onSubmittedWorkDone(); child.close(); }
    catch (settlement) {
      throw Object.assign(new AggregateError([error, settlement], 'Replay settlement failed.'), { unsettledAttempt: child });
    }
    throw error;
  }
}
