const engines = ['pooled', 'doppler'];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function validateResidentPlan(contract, method, prompts) {
  const shared = contract.shared;
  if (contract.id !== 'qwen35-canonical-f32-f32kv' || shared.kvDtype !== 'f32'
      || shared.maxNewTokens !== 64 || shared.warmupRuns !== 1 || shared.pairsPerPrompt !== 5) {
    throw new Error('Resident runner requires the unchanged F32-KV contract');
  }
  if (method.promptCount !== 4 || method.warmupsPerEnginePerPrompt !== shared.warmupRuns
      || method.pairsPerPrompt !== shared.pairsPerPrompt || method.totalPairs !== 20
      || method.pairOrder !== 'alternate-global-pair-index-pooled-first-even'
      || !same(method.engines, engines) || method.outliersRemoved !== 0 || method.sampling?.repetitionPenalty !== 1) throw new Error('Unsupported resident pairing method');
  if (!Array.isArray(prompts) || prompts.length !== method.promptCount
      || new Set(prompts.map(prompt => prompt.id)).size !== prompts.length) throw new Error('Four distinct frozen prompts required');
  for (const prompt of prompts) {
    if (typeof prompt.id !== 'string' || !/^[a-z0-9-]+$/.test(prompt.id) || typeof prompt.text !== 'string' || typeof prompt.expectedText !== 'string'
        || !Array.isArray(prompt.promptIds) || !prompt.promptIds.length
        || !Array.isArray(prompt.expectedTokenIds) || prompt.expectedTokenIds.length !== shared.maxNewTokens
        || ![...prompt.promptIds, ...prompt.expectedTokenIds].every(id => Number.isSafeInteger(id) && id >= 0)
        || prompt.promptIds.length + shared.maxNewTokens > shared.maxSeqLen
        || prompt.expectedTokenIds.slice(0, -1).some(id => shared.eosTokenIds.includes(id))) {
      throw new Error(`Invalid full-budget reference prompt ${prompt.id}`);
    }
  }
}

export function residentSchedule(method, prompts) {
  const schedule = [];
  let pair = 0;
  for (let promptIndex = 0; promptIndex < prompts.length; promptIndex++) {
    const promptId = prompts[promptIndex].id;
    for (const engine of promptIndex % 2 ? engines.toReversed() : engines) {
      schedule.push({ promptId, engine, warmup: true, pair: null });
    }
    for (let index = 0; index < method.pairsPerPrompt; index++, pair++) {
      for (const engine of pair % 2 ? engines.toReversed() : engines) {
        schedule.push({ promptId, engine, warmup: false, pair });
      }
    }
  }
  return schedule.map((request, sequence) => ({ ...request, sequence }));
}

export function validateResidentOutput(result, prompt, contract) {
  if (!same(result.promptIds, prompt.promptIds)) throw new Error('Actual prompt token IDs differ');
  if (!same(result.tokenIds, prompt.expectedTokenIds)) throw new Error('Generated token IDs differ from pinned reference');
  if (typeof result.text !== 'string' || result.text !== prompt.expectedText) throw new Error('Decoded response text differs from pinned reference');
  const eos = contract.shared.eosTokenIds;
  if (result.tokenIds.length !== contract.shared.maxNewTokens
      || result.tokenIds.slice(0, -1).some(id => eos.includes(id))) throw new Error('Full natural token budget was not completed');
  const stopReason = eos.includes(result.tokenIds.at(-1)) ? 'eos' : 'max-tokens';
  if (result.stopReason !== stopReason) throw new Error('Native stop reason differs from shared EOS rules');
  const times = result.deliveryMs;
  if (!Number.isFinite(result.completeMs) || result.completeMs <= 0 || !Array.isArray(times)
      || times.length !== result.tokenIds.length || times.some((time, i) => !Number.isFinite(time)
        || time < 0 || time > result.completeMs || (i && time < times[i - 1]))
      || result.firstTokenMs !== times[0]) throw new Error('Invalid measured request or delivery clock');
}

// record() must finish durable persistence before the next request can start.
export async function runResidentPairs({ sessions, contract, method, prompts, record }) {
  validateResidentPlan(contract, method, prompts);
  const attempts = [];
  let primaryError;
  try {
    for (const engine of engines) await sessions[engine].ready();
    for (const request of residentSchedule(method, prompts)) {
      const prompt = prompts.find(value => value.id === request.promptId);
      const attempt = { ...request, status: 'started' };
      let failure;
      try {
        await sessions[request.engine].prepare(prompt);
        Object.assign(attempt, await sessions[request.engine].request(prompt));
        validateResidentOutput(attempt, prompt, contract);
        attempt.status = 'passed';
      } catch (error) {
        Object.assign(attempt, error.partialRequest ?? {});
        attempt.status = 'failed';
        attempt.error = String(error.stack || error);
        failure = error;
      }
      attempts.push(attempt);
      await record(attempt);
      if (failure) throw failure;
    }
    return attempts;
  } catch (error) {
    primaryError = error;
    error.attempts = attempts;
    throw error;
  } finally {
    const cleanup = await Promise.allSettled(engines.map(engine => sessions[engine].close()));
    const failures = cleanup.flatMap((result, index) => result.status === 'rejected'
      ? [{ engine: engines[index], error: String(result.reason) }] : []);
    if (failures.length) {
      await record({ type: 'cleanup-failures', failures });
      if (!primaryError) throw new Error(`Resident session cleanup failed: ${JSON.stringify(failures)}`);
    }
  }
}
