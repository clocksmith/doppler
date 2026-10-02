import { initDevice, getDevice } from '../gpu/device.js';
import { setDeviceMemoryBudget, getDeviceMemorySnapshot } from '../memory/device-budget.js';

/** Explicit host allocation limit; shared by all model sessions on this device.
 * @type {import('./resident-partitions.js').configureDeviceMemoryBudget} */
export async function configureDeviceMemoryBudget({ maxBytes }) {
  return setDeviceMemoryBudget(await initDevice(), maxBytes);
}

export function inspectDeviceMemory() {
  return getDeviceMemorySnapshot(getDevice());
}

/** @type {import('./resident-partitions.js').createResidentPartitionFactory} */
export function createResidentPartitionFactory({ openCapsule, capsuleOptions }) {
  if (typeof openCapsule !== 'function' || !capsuleOptions || typeof capsuleOptions !== 'object') {
    throw new Error('Resident partitions require the normal Capsule opener and explicit host trust options.');
  }
  const hostOptions = { ...capsuleOptions };
  return Object.freeze({
    /** @param {import('./resident-partitions.js').ResidentPartitionOpenOptions} options */
    async openResidentPartition({ model, plan, planId, index, participantId, limits, signal }) {
    signal.throwIfAborted();
    if (!model.capsule || !model.generation) throw new Error('Resident model requires a Capsule source and explicit generation settings.');
    const allocation = structuredClone({ model: { id: model.id, identity: model.identity }, plan, planId,
      index, participantId, limits, generation: model.generation });
    const session = await openCapsule(structuredClone(model.capsule), { ...hostOptions, signal, residentPartition: allocation });
    try {
      signal.throwIfAborted();
      if (!session.residentPartition) throw new Error('Capsule opener does not implement resident partitions.');
      return session.residentPartition;
    } catch (error) {
      try { await session.close(); }
      catch (cleanupError) { throw new AggregateError([error, cleanupError], 'Resident preparation and cleanup failed.'); }
      throw error;
    }
  } });
}

/** Explicit pinned-manifest development lane. Unlike createResidentPartitionFactory,
 * this does not assert signed Capsule qualification. Hosts supply already verified
 * storage and an exact manifest identity; no model identity is inferred from a URL.
 * @type {import('./resident-partitions.js').createManifestResidentPartitionFactory}
 */
export function createManifestResidentPartitionFactory({ manifest, manifestIdentity, runtimeConfig, createStorage }) {
  const source = structuredClone(manifest), config = structuredClone(runtimeConfig);
  if (!/^sha256:[a-f0-9]{64}$/.test(manifestIdentity) || typeof createStorage !== 'function'
    || !config || !source?.modelId) throw new Error('Pinned manifest, verified storage and explicit runtime configuration required.');
  return Object.freeze({
    async openResidentPartition(input) {
      const [{ createPipeline, InferencePipeline }, { createModelHandle }, { createResidentPartitionSession }, { resolveResidentPartitionAllocation }] = await Promise.all([
        import('../inference/pipelines/text.js'), import('./model-host/model-session.js'),
        import('../inference/pipelines/text/resident-partition.js'), import('../inference/pipelines/text/resident-partition-contract.js'),
      ]);
      input.signal.throwIfAborted();
      const { signal, ...allocationInput } = input;
      const allocation = resolveResidentPartitionAllocation(source, manifestIdentity, { ...allocationInput, generation: input.model.generation });
      const storage = await createStorage(input);
      /** @type {ReturnType<typeof createModelHandle> | null} */
      let handle = null;
      try {
        const pipeline = await createPipeline(source, { runtimeConfig: config, storage: {
          ...storage, loadShardRange: storage.loadShardRange ?? undefined, streamShardRange: storage.streamShardRange ?? undefined,
          loadTokenizerJson: storage.loadTokenizerJson ?? undefined, loadTokenizerModel: storage.loadTokenizerModel ?? undefined,
          loadAuxiliaryFile: storage.loadAuxiliaryFile ?? undefined, loadTensorsJson: storage.loadTensorsJson ?? undefined,
          close: storage.close ?? undefined },
          partition: { plan: allocation.plan, index: allocation.index } });
        if (!(pipeline instanceof InferencePipeline)) {
          await pipeline.unload();
          throw new Error('Resident partitions require a text inference pipeline.');
        }
        const opened = createModelHandle(pipeline, { modelId: source.modelId, manifestHash: manifestIdentity.slice(7) });
        handle = opened;
        input.signal.throwIfAborted();
        /** @type {import('../inference/pipelines/text/resident-partition-contract.js').ResidentPartitionTokenPorts} */
        const tokens = {
          tokenize: (prompt, options) => opened.advanced.tokenizePrompt(prompt, options),
          createIncrementalDecoder: () => opened.advanced.createIncrementalDecoder(),
          getTokenContract: () => {
            const special = opened.advanced.getSpecialTokens();
            return { padTokenId: Number.isInteger(special.pad) ? special.pad : null,
              eosTokenId: Number.isInteger(special.eos) ? special.eos : null,
              stopTokenIds: opened.advanced.getStopTokenIds() };
          },
        };
        return await createResidentPartitionSession(pipeline, allocation, tokens, async () => {
          try { await opened.unload(); } finally { await storage.close?.(); }
        });
      } catch (error) {
        try { await handle?.unload(); } finally { await storage.close?.(); }
        throw error;
      }
    },
  });
}
