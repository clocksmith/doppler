// Interchangeable scoring engines with one contract:
//   scoreChoices({ prompt, choices, maxSeqLen }, { signal }) -> { choices: [{ id, tokenId, logit }] }
// Both reject prompts over maxSeqLen instead of truncating them.
import { createModelInstallation } from './installation.js';

// Doppler: a signed Capsule installed into private browser storage, so later
// openings verify retained bytes and work offline.
export async function openDopplerEngine({ descriptor, baseUrl, mode = 'install-or-open', signal, onProgress }) {
  const [host, runtime, storage] = await Promise.all([
    import('doppler-gpu/host'),
    import('doppler-gpu'),
    import('doppler-gpu/tooling/storage'),
  ]);
  const capsuleUrl = new URL(descriptor.capsuleUrl, baseUrl).href;
  const capsule = mode === 'open-retained' ? null : await (await fetch(capsuleUrl, { signal })).json();
  const store = storage.createOpfsStore(descriptor.storage);
  await store.openModel(descriptor.storageId);
  const source = runtime.createFetchCapsuleArtifactStore(capsuleUrl);
  const installation = createModelInstallation({
    store,
    openCapsule: host.openCapsule,
    fetchArtifact: (artifact, control) => source.readArtifact(artifact, control),
    authorizeRecord: (record) => record.capsule?.semanticRoot === descriptor.semanticRoot
      && JSON.stringify(record.options.trustedSigners) === JSON.stringify(descriptor.openOptions.trustedSigners)
      && JSON.stringify(record.options.acceptedTargetPlanDigests) === JSON.stringify(descriptor.openOptions.acceptedTargetPlanDigests),
    withLock: (task) => navigator.locks.request(`issue-triage-${descriptor.storageId}`, task),
  });
  const record = capsule && { capsule, options: { ...descriptor.openOptions, requiredOperations: ['scoreChoices'] } };
  const controls = { signal, observer: onProgress ? { observe: onProgress } : undefined };
  let opened;
  if (mode === 'open-retained') opened = await installation.openRetained(controls);
  else if (mode === 'install') opened = await installation.install(record, controls);
  else {
    try {
      opened = await installation.openRetained(controls);
    } catch (error) {
      if (!/No completed model installation/.test(error.message)) throw error;
      const fresh = await (await fetch(capsuleUrl, { signal })).json();
      opened = await installation.install({ capsule: fresh, options: { ...descriptor.openOptions, requiredOperations: ['scoreChoices'] } }, controls);
    }
  }
  const { session, observations } = opened;
  return {
    name: 'doppler',
    observations,
    async scoreChoices(request, options = {}) {
      return host.validateChoiceScoringResult(request, await session.scoreChoices(request, options));
    },
    close: () => session.close(),
  };
}

// Transformers.js on the same source checkpoint, reading the same answer-token logits.
// Its default browser cache retains downloaded model files.
export async function openTransformersEngine({ transformers, modelId, dtype = 'fp16', device = 'webgpu', signal }) {
  const { AutoTokenizer, AutoModelForCausalLM } = transformers;
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(modelId),
    AutoModelForCausalLM.from_pretrained(modelId, { device, dtype }),
  ]);
  signal?.throwIfAborted();
  const encode = (text) => Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
  return {
    name: 'transformersjs',
    observations: [],
    async scoreChoices({ prompt, choices, maxSeqLen }, { signal: requestSignal } = {}) {
      requestSignal?.throwIfAborted();
      const promptIds = encode(prompt);
      if (promptIds.length === 0 || promptIds.length >= maxSeqLen) {
        throw new Error('Choice scoring prompt is empty, invalid or exceeds maxSeqLen.');
      }
      const tokenIds = choices.map((choice) => {
        const ids = encode(prompt + choice.label);
        const appended = ids.length - promptIds.length;
        if (appended !== 1 || ids.slice(0, promptIds.length).some((id, i) => id !== promptIds[i])) {
          throw new Error(`Choice label "${choice.id}" must append exactly one token at the answer position.`);
        }
        return ids.at(-1);
      });
      const inputs = tokenizer(prompt, { add_special_tokens: false, padding: false, truncation: false });
      // Transformers.js has no cancellation for a submitted forward pass; the
      // signal is honored only before and after it.
      const output = await model(inputs);
      requestSignal?.throwIfAborted();
      // fp16 logits arrive as raw Uint16 bits where Float16Array is missing (Node 22).
      const logits = output.logits.type === 'float16' ? output.logits.to('float32') : output.logits;
      const { data, dims } = logits;
      const vocab = dims.at(-1);
      const offset = (dims.at(-2) - 1) * vocab;
      const scored = choices.map((choice, i) => ({
        id: choice.id,
        label: choice.label,
        tokenId: tokenIds[i],
        logit: Number(data[offset + tokenIds[i]]),
      }));
      output.logits.dispose?.();
      if (logits !== output.logits) logits.dispose?.();
      const selected = scored.reduce((best, choice) => (choice.logit > best.logit ? choice : best));
      return { choices: scored, selectedId: selected.id, promptTokenCount: promptIds.length };
    },
    async close() {
      await model.dispose?.();
    },
  };
}
