import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';

/** @param {unknown} value @param {string} label */
function requireText(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`LoRA round: ${label} is required.`);
}

/** @param {import('./lora-training-round.js').LoRATrainingRoundInput} input */
export function createLoRATrainingRound(input) {
  const body = {
    ...structuredClone(input),
    schema: /** @type {const} */ ('doppler.lora-training-round/v1'),
    checkpointHash: computeCanonicalSha256(input.checkpoint),
  };
  const round = { ...body, roundHash: computeCanonicalSha256(body) };
  validateLoRATrainingRound(round);
  return round;
}

/** @param {import('./lora-training-round.js').LoRATrainingRound} round */
export function validateLoRATrainingRound(round) {
  if (round?.schema !== 'doppler.lora-training-round/v1') throw new Error('LoRA round: unsupported schema.');
  requireText(round.jobId, 'jobId');
  requireText(round.roundId, 'roundId');
  for (const hash of [round.baseManifestHash, round.trainingConfigHash, round.checkpointHash, round.roundHash]) {
    if (!/^sha256:[a-f0-9]{64}$/.test(hash)) throw new Error('LoRA round: invalid identity digest.');
  }
  if (!Array.isArray(round.assignments) || round.assignments.length === 0) {
    throw new Error('LoRA round: assignments are required.');
  }
  const ids = new Set();
  let totalTokens = 0;
  for (const assignment of round.assignments) {
    requireText(assignment.participantId, 'participantId');
    if (ids.has(assignment.participantId)) throw new Error('LoRA round: duplicate participant.');
    ids.add(assignment.participantId);
    if (!/^sha256:[a-f0-9]{64}$/.test(assignment.sampleHash)
      || !Number.isSafeInteger(assignment.supervisedTokenCount) || assignment.supervisedTokenCount < 1) {
      throw new Error('LoRA round: invalid sample assignment.');
    }
    totalTokens += assignment.supervisedTokenCount;
  }
  if (!Number.isSafeInteger(totalTokens)) throw new Error('LoRA round: supervised token count overflow.');
  if (computeCanonicalSha256(round.checkpoint) !== round.checkpointHash) {
    throw new Error('LoRA round: checkpoint identity mismatch.');
  }
  const { roundHash, ...body } = round;
  if (computeCanonicalSha256(body) !== roundHash) throw new Error('LoRA round: contract identity mismatch.');
}

/**
 * Require the entire declared round before an optimizer update. Arrival order
 * never changes summation order, and retries cannot count a contributor twice.
 * @param {import('./lora-training-round.js').LoRATrainingRound} round
 * @param {import('./lora-training-round.js').LoRAGradientContribution[]} contributions
 */
export function validateLoRAContributions(round, contributions) {
  validateLoRATrainingRound(round);
  if (!Array.isArray(contributions) || contributions.length !== round.assignments.length) {
    throw new Error('LoRA round: incomplete contribution set.');
  }
  const byId = new Map();
  for (const contribution of contributions) {
    if (contribution?.schema !== 'doppler.lora-gradient-contribution/v1'
      || contribution.roundHash !== round.roundHash) throw new Error('LoRA round: stale or foreign contribution.');
    if (byId.has(contribution.participantId)) throw new Error('LoRA round: duplicate contribution.');
    if (!Number.isFinite(contribution.loss)) throw new Error('LoRA round: non-finite loss.');
    byId.set(contribution.participantId, contribution);
  }
  return round.assignments.map((assignment) => {
    const contribution = byId.get(assignment.participantId);
    if (!contribution || contribution.sampleHash !== assignment.sampleHash
      || contribution.supervisedTokenCount !== assignment.supervisedTokenCount) {
      throw new Error('LoRA round: contribution does not match its assignment.');
    }
    return contribution;
  });
}

/** @param {string} participantId @param {import('./lora-training-round.js').LoRATrainingSample} sample */
export function describeLoRATrainingSample(participantId, sample) {
  requireText(participantId, 'participantId');
  if (!Array.isArray(sample?.inputIds) || !Array.isArray(sample?.targetIds)
    || sample.inputIds.length === 0 || sample.inputIds.length !== sample.targetIds.length
    || !Number.isSafeInteger(sample.supervisedTokenCount) || sample.supervisedTokenCount < 1
    || sample.supervisedTokenCount > sample.targetIds.length) {
    throw new Error('LoRA round: invalid training sample.');
  }
  return { participantId, sampleHash: computeCanonicalSha256(sample), supervisedTokenCount: sample.supervisedTokenCount };
}
