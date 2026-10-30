// Host request timing only. Output checks run after each measured interval.
export async function measurePooledRequests(engine, device, promptIds, contract, timing, now = () => performance.now()) {
  if (timing?.diagnosticOnly !== true) throw new Error('Pilot timing is diagnostic only');
  for (const [name, minimum] of [['warmupRuns', 0], ['timedRuns', 1]]) {
    if (!Number.isSafeInteger(timing[name]) || timing[name] < minimum) throw new Error(`Invalid timing.${name}`);
  }
  if (!Array.isArray(promptIds) || !promptIds.length || !promptIds.every(Number.isSafeInteger)) throw new Error('Prompt token IDs are required');
  const budget = contract.shared.maxNewTokens, eos = contract.shared.eosTokenIds;
  if (!Number.isSafeInteger(budget) || budget <= 0 || !Array.isArray(eos)) throw new Error('Invalid shared token budget or stop rules');
  if (!Array.isArray(timing.expectedTokenIds) || !timing.expectedTokenIds.length) throw new Error('Pinned expected output tokens are required');
  const samples = [];
  try {
    for (let run = 0; run < timing.warmupRuns + timing.timedRuns; run++) {
      engine.reset();
      await device.queue.onSubmittedWorkDone();
      const started = now();
      await engine.prefillTokens(promptIds.slice(0, -1));
      let token = (await engine.forwardTokenIds(promptIds.at(-1))).ids[0];
      const firstTokenMs = now() - started, tokenIds = [token];
      while (tokenIds.length < budget && !eos.includes(token)) {
        token = (await engine.forwardTokenIds(token)).ids[0];
        tokenIds.push(token);
      }
      await device.queue.onSubmittedWorkDone();
      const completeMs = now() - started;
      if (!Number.isFinite(completeMs) || completeMs <= 0 || firstTokenMs < 0 || firstTokenMs > completeMs) throw new Error('Invalid request clock');
      const valid = tokenIds.length === timing.expectedTokenIds.length
        && tokenIds.every((id, i) => id === timing.expectedTokenIds[i]);
      samples.push({ run, warmup: run < timing.warmupRuns, firstTokenMs, completeMs, tokenIds, valid,
        stopReason: eos.includes(tokenIds.at(-1)) ? 'eos' : 'max-tokens' });
      if (!valid) throw new Error('Timed request output differs from the pinned numerical reference');
    }
  } catch (error) {
    error.completedSamples = samples;
    throw error;
  }
  return samples;
}
