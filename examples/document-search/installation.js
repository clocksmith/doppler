// This adapter persists bytes through Doppler's existing storage backend.
// Application trust, plan approval and release checkpoints remain explicit inputs.
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const decode = bytes => JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
async function digest(bytes) {
  return 'sha256:' + Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    value => value.toString(16).padStart(2, '0')).join('');
}

export function createDocumentModelInstallation({ store, fetchArtifact, openCapsule, authorizeRecord, withLock }) {
  if (typeof authorizeRecord !== 'function' || typeof withLock !== 'function') {
    throw new Error('Application record authorization and an exclusive installation lock are required.');
  }
  function key(artifact) {
    if (!/^sha256:[a-f0-9]{64}$/.test(artifact.hash) || !Number.isSafeInteger(artifact.sizeBytes) || artifact.sizeBytes < 0) {
      throw new Error('A signed SHA-256 artifact descriptor is required.');
    }
    return 'artifacts/' + artifact.hash.slice(7);
  }
  async function verify(artifact, bytes) {
    if (bytes.byteLength !== artifact.sizeBytes || await digest(bytes) !== artifact.hash) {
      throw new Error(`Stored artifact integrity failed: ${artifact.artifactId}`);
    }
    return bytes;
  }
  async function readOrMissing(filename) {
    try { return await store.readFile(filename); }
    catch (error) { if (error.name === 'NotFoundError') return null; throw error; }
  }
  function artifactStore(acquire, observations, repair) {
    const reads = new Set();
    async function readArtifact(artifact, control = {}) {
      control.signal?.throwIfAborted();
      const filename = key(artifact);
      let started = performance.now();
      const retained = await readOrMissing(filename);
      const storageReadMs = performance.now() - started;
      if (retained !== null) {
        let valid = true;
        started = performance.now();
        try { await verify(artifact, retained); }
        catch (error) { if (!acquire || !repair) throw error; valid = false; }
        if (valid) {
          observations.push({ artifactId: artifact.artifactId, source: 'storage', bytes: retained.byteLength,
            storageReadMs, verificationMs: performance.now() - started });
          control.signal?.throwIfAborted();
          return retained;
        }
        control.signal?.throwIfAborted();
        await store.deleteFile(filename);
        observations.push({ artifactId: artifact.artifactId, action: 'removed-damaged-artifact' });
      }
      if (!acquire) throw new Error(`Offline installation is incomplete: ${artifact.artifactId}`);
      started = performance.now();
      const acquired = await fetchArtifact(artifact, control);
      const acquisitionMs = performance.now() - started;
      started = performance.now();
      const bytes = await verify(artifact, acquired);
      const verificationMs = performance.now() - started;
      control.signal?.throwIfAborted();
      started = performance.now();
      try { await store.writeFile(filename, bytes); }
      catch (error) {
        // This artifact was absent under the exclusive installation lock.
        try { await store.deleteFile(filename); }
        catch (cleanupError) { throw new AggregateError([error, cleanupError], error.message, { cause: error }); }
        throw error;
      }
      observations.push({ artifactId: artifact.artifactId, source: 'network', bytes: bytes.byteLength,
        storageReadMs, acquisitionMs, verificationMs, storageWriteMs: performance.now() - started });
      control.signal?.throwIfAborted();
      return bytes;
    }
    return {
      readArtifact(artifact, control) {
        const task = readArtifact(artifact, control);
        reads.add(task);
        const remove = () => reads.delete(task);
        task.then(remove, remove);
        return task;
      },
      async settle() { await Promise.allSettled(reads); },
    };
  }
  async function checkpoint() {
    const bytes = await readOrMissing('release-checkpoint.json');
    return bytes === null ? null : decode(bytes);
  }
  async function optionsFor(record, observations, acquire, controls) {
    if (await authorizeRecord(record) !== true) throw new Error('Model installation is not application-authorized.');
    const { repairDamagedArtifacts, ...runtimeControls } = controls;
    const options = { ...record.options, ...runtimeControls,
      artifactStore: artifactStore(acquire, observations, repairDamagedArtifacts === true) };
    if (options.releasePolicy) {
      options.releasePolicy = { ...options.releasePolicy, now: new Date().toISOString() };
      const prior = await checkpoint();
      if (prior && prior.sequence > options.releasePolicy.checkpoint.sequence) {
        options.releasePolicy = { ...options.releasePolicy, checkpoint: prior,
          minimumSequence: Math.max(prior.sequence, options.releasePolicy.minimumSequence) };
      } else if (prior && prior.sequence === options.releasePolicy.checkpoint.sequence
        && prior.digest !== options.releasePolicy.checkpoint.digest) throw new Error('Release checkpoint fork.');
      options.persistReleaseCheckpoint = async next => {
        const current = await checkpoint();
        if (current && (next.sequence < current.sequence || (next.sequence === current.sequence && next.digest !== current.digest))) {
          throw new Error('Release checkpoint rollback or fork.');
        }
        await store.writeFile('release-checkpoint.json', encode(next));
      };
    }
    return options;
  }
  return {
    install(record, controls = {}) { return withLock(async () => {
      const observations = [];
      let session;
      let options;
      try {
        options = await optionsFor(record, observations, true, controls);
        session = await openCapsule(record.capsule, options);
        controls.signal?.throwIfAborted();
        await store.writeFile('installation.json', encode(record));
        return { session, observations };
      } catch (error) {
        // Cancellation may finish runtime acquisition before a storage write settles.
        // Keep the installation lock until its outstanding reads and writes finish.
        await options?.artifactStore.settle();
        try { await session?.close(); } catch (cleanupError) {
          throw new AggregateError([error, cleanupError], error.message, { cause: error });
        }
        throw error;
      }
    }); },
    openRetained(controls = {}) { return withLock(async () => {
      const bytes = await readOrMissing('installation.json');
      if (bytes === null) throw new Error('No completed model installation.');
      const record = decode(bytes);
      const observations = [];
      const options = await optionsFor(record, observations, false, controls);
      try { return { record, observations, session: await openCapsule(record.capsule, options) }; }
      catch (error) { await options.artifactStore.settle(); throw error; }
    }); },
    removeDamagedArtifact(artifact) { return withLock(async () => {
      const filename = key(artifact);
      const bytes = await readOrMissing(filename);
      if (bytes === null) return false;
      try { await verify(artifact, bytes); return false; }
      catch { return store.deleteFile(filename); }
    }); },
  };
}
