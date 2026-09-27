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
