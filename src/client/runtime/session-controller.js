/**
 * Doppler SessionController: Controls KV cache lifecycle and token iteration loops.
 *
 * @module client/runtime/session-controller
 */

/**
 * Creates a session controller for managing model generation cycles.
 *
 * @param {object} commandExecutor Injected command executor
 * @param {object} resourceBinder Injected resource binder
 * @returns {object} Session controller instance
 */
export function createSessionController(commandExecutor, resourceBinder) {
  if (!commandExecutor || !resourceBinder) {
    throw new Error('createSessionController requires commandExecutor and resourceBinder.');
  }

  return {
    /**
     * Executes a complete forward generation cycle from a selected TargetPlan.
     *
     * @param {object} targetPlan The selected TargetPlan
     * @param {object} options Generation options ({ promptTokens, maxTokens, signal })
     * @returns {AsyncGenerator<number, void, void>} Generated token stream
     */
    async *generateTokens(targetPlan, options = {}) {
      const { promptTokens = [], maxTokens = 16, signal = null } = options;
      if (signal?.aborted) {
        throw new Error('Generation aborted before prefill.');
      }

      // 1. Bind resources for prompt length
      resourceBinder.bindSlots(targetPlan.memoryLayout, {
        seqLen: promptTokens.length,
      });

      // 2. Prefill phase execution
      await commandExecutor.executePhase('prefill', targetPlan.phases?.prefill || [], { signal });

      // 3. Decode iteration loop
      for (let step = 0; step < maxTokens; step += 1) {
        if (signal?.aborted) {
          throw new Error('Generation aborted during decode.');
        }

        await commandExecutor.executePhase('decode', targetPlan.phases?.decode || [], { signal });
        const dummyNextToken = (promptTokens[promptTokens.length - 1] || 0) + step + 1;
        yield dummyNextToken;
      }

      // 4. Release transient memory
      resourceBinder.releaseTransient();
    },
  };
}
