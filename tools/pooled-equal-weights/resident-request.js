// Timing adapters delegate engine generation; validation and harness-only token
// diagnostics happen after the final queue fence and end timestamp.
export async function measureDopplerRequest({ pipeline, device, runTextInference, runtimeConfig,
  prompt, contract, method, now = () => performance.now() }) {
  let started, completeMs;
  const tokenIds = [], deliveryMs = [], callbacks = [], chunks = [];
  const generate = pipeline.generate.bind(pipeline);
  const facade = new Proxy(pipeline, { get(target, key) {
    if (key !== 'generate') {
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    }
    return async function* (text, options) {
      const nativeOptions = { ...options, inputIds: [...prompt.promptIds], onToken(id, chunk) {
        tokenIds.push(id);
        deliveryMs.push(now() - started);
        callbacks.push([id, chunk]);
      } };
      started = now();
      for await (const chunk of generate(text, nativeOptions)) {
        chunks.push(chunk);
        yield chunk;
      }
      await device.queue.onSubmittedWorkDone();
      completeMs = now() - started;
      // runTextInference's fallbackText decoding is evidence construction, not
      // generation. Preserve it outside both the common clock and native loop.
      for (const args of callbacks) options.onToken?.(...args);
    };
  } });
  try {
    const result = await runTextInference(facade, runtimeConfig, {
      prompt: contract.promptFormat.before + prompt.text + contract.promptFormat.after,
      useChatTemplate: false, maxTokens: contract.shared.maxNewTokens, benchmark: true,
      seed: contract.shared.seed, captureReferenceLogits: false, referenceTranscript: false,
      diagnostics: { enabled: false },
      sampling: { temperature: contract.shared.temperature, topK: contract.shared.topK,
        topP: contract.shared.topP, repetitionPenalty: method.sampling.repetitionPenalty },
    });
    return { promptIds: result.promptTokenIds, tokenIds, deliveryMs, firstTokenMs: deliveryMs[0], completeMs,
      text: chunks.join(''), generationConfig: result.generationConfig, stopReason: contract.shared.eosTokenIds.includes(tokenIds.at(-1)) ? 'eos' : 'max-tokens',
      nativeApi: 'pipeline.generate(formattedString, { inputIds, onToken }) via runTextInference',
      nativeStopReason: pipeline.getStats?.()?.stopReason ?? null, diagnosticsOutsideInterval: true };
  } catch (error) {
    error.partialRequest = { tokenIds, deliveryMs, firstTokenMs: deliveryMs[0] ?? null,
      completeMs: completeMs ?? null, elapsedAtFailureMs: started === undefined ? null : now() - started };
    throw error;
  }
}

export async function measurePooledRequest({ engine, device, tokenizer, prompt, contract, now = () => performance.now() }) {
  const tokenIds = [], deliveryMs = [];
  const prefix = prompt.promptIds.slice(0, -1), last = prompt.promptIds.at(-1);
  const started = now();
  try {
    await engine.prefillTokens(prefix);
    let input = last;
    while (tokenIds.length < contract.shared.maxNewTokens) {
      const token = (await engine.forwardTokenIds(input)).ids[0];
      tokenIds.push(token);
      deliveryMs.push(now() - started);
      if (contract.shared.eosTokenIds.includes(token)) break;
      input = token;
    }
    // Native buffered text response, not repeated full-prefix decoding.
    const textIds = contract.shared.eosTokenIds.includes(tokenIds.at(-1)) ? tokenIds.slice(0, -1) : tokenIds;
    const text = tokenizer.decode(textIds);
    await device.queue.onSubmittedWorkDone();
    const completeMs = now() - started;
    return { promptIds: prompt.promptIds, tokenIds, deliveryMs, firstTokenMs: deliveryMs[0], completeMs, text,
      stopReason: contract.shared.eosTokenIds.includes(tokenIds.at(-1)) ? 'eos' : 'max-tokens',
      nativeApi: 'prefillTokens + forwardTokenIds; native tokenizer.decode once before completion' };
  } catch (error) {
    error.partialRequest = { tokenIds, deliveryMs, firstTokenMs: deliveryMs[0] ?? null,
      completeMs: null, elapsedAtFailureMs: now() - started };
    throw error;
  }
}
