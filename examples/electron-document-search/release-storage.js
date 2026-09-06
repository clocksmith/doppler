import fs from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { validateElectronReleaseState } from 'doppler-gpu/electron';
import { verifyPackReleaseEvents, PackReleaseStateError } from 'doppler-gpu/pack';

function checkpoint(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || Object.keys(value).length !== 2 || !Object.hasOwn(value, 'digest')
    || !Number.isSafeInteger(value.sequence) || value.sequence < 0
    || (value.sequence === 0 ? value.digest !== null : !/^sha256:[0-9a-f]{64}$/u.test(value.digest))) {
    throw new Error('Document search release checkpoint requires an exact sequence/digest record.');
  }
  return { sequence: value.sequence, digest: value.digest };
}

// Main-process example for a private application directory on a local filesystem
// supporting exclusive creation, atomic rename and directory fsync. A leftover
// crash lock fails closed; never delete it merely because it appears old.
function fileStore(filename, validate) {
  if (typeof filename !== 'string' || !path.isAbsolute(filename)) {
    throw new Error('Document search release store requires an absolute filename.');
  }
  if (typeof constants.O_NOFOLLOW !== 'number' || constants.O_NOFOLLOW === 0) {
    throw new Error('Document search release store requires no-follow file support.');
  }
  const lockPath = `${filename}.lock`;
  const directory = path.dirname(filename);

  async function syncDirectory() {
    const parent = await fs.open(directory, 'r');
    let failed = false;
    try { await parent.sync(); }
    catch (error) { failed = true; throw error; }
    finally { try { await parent.close(); } catch (error) { if (!failed) throw error; } }
  }

  async function load() {
    let handle;
    try {
      handle = await fs.open(filename, constants.O_RDONLY | constants.O_NOFOLLOW);
    } catch (error) {
      if (error.code === 'ENOENT') return null;
      throw error;
    }
    let failed = false;
    try {
      if (!(await handle.stat()).isFile()) throw new Error('Release state must be a regular file.');
      const value = validate(JSON.parse(await handle.readFile('utf8')));
      // A previous rename may be visible even if its durability barrier failed.
      // Re-establish durability before treating an identical record as committed.
      await handle.sync();
      await syncDirectory();
      return value;
    } catch (error) { failed = true; throw error; }
    finally { try { await handle.close(); } catch (error) { if (!failed) throw error; } }
  }

  async function compareAndSwap(expectedSequence, nextState) {
    const next = validate(structuredClone(nextState));
    if (!Number.isSafeInteger(expectedSequence) || expectedSequence < 0
      || !Number.isSafeInteger(next.sequence) || next.sequence <= expectedSequence) {
      throw new Error('Document search release store writes must advance a valid sequence.');
    }
    let lock;
    try { lock = await fs.open(lockPath, 'wx', 0o600); }
    catch (error) { if (error.code === 'EEXIST') return false; throw error; }
    let temporary = null;
    let failed = false;
    try {
      const current = await load();
      if ((current?.sequence ?? 0) !== expectedSequence) return false;
      temporary = `${filename}.${randomUUID()}.tmp`;
      const output = await fs.open(temporary, 'wx', 0o600);
      let writeFailed = false;
      try {
        await output.writeFile(`${JSON.stringify(next)}\n`, 'utf8');
        await output.sync();
      } catch (error) { writeFailed = true; throw error; }
      finally { try { await output.close(); } catch (error) { if (!writeFailed) throw error; } }
      await fs.rename(temporary, filename);
      temporary = null;
      await syncDirectory();
      return true;
    } catch (error) {
      failed = true;
      throw error;
    } finally {
      const errors = [];
      if (temporary) {
        try { await fs.unlink(temporary); } catch (error) { if (error.code !== 'ENOENT') errors.push(error); }
      }
      try { await lock.close(); } catch (error) { errors.push(error); }
      try { await fs.unlink(lockPath); } catch (error) { errors.push(error); }
      if (!failed && errors.length) throw errors[0];
    }
  }

  return Object.freeze({ load, compareAndSwap });
}

export function createDocumentSearchReleaseStore(filename) {
  return fileStore(filename, validateElectronReleaseState);
}

export function createDocumentSearchCheckpointStore(filename) {
  return fileStore(filename, checkpoint);
}

// Prepare this context again for each open, using a store dedicated to the
// application's authorized release stream. Downloading a new event does not
// activate a Pack. The caller still selects and authorizes the executable.
export async function prepareDocumentSearchReleaseOptions({
  pack, releaseEvents, releaseTrustedSigners, checkpointStore, minimumSequence, now, retainedLocalUse,
}) {
  if (pack?.schema !== 'doppler.pack/v3') throw new Error('Release checkpoint preparation requires Pack v3.');
  if (typeof checkpointStore?.load !== 'function' || typeof checkpointStore?.compareAndSwap !== 'function') {
    throw new Error('Release checkpoint preparation requires a durable compare-and-swap store.');
  }
  const events = structuredClone(releaseEvents);
  const signers = structuredClone(releaseTrustedSigners);
  const decision = structuredClone(retainedLocalUse);
  const previous = checkpoint(await checkpointStore.load() ?? { sequence: 0, digest: null });
  const policy = { now, minimumSequence, checkpoint: previous,
    ...(decision === undefined ? {} : { retainedLocalUse: decision }) };
  let expected;

  async function persistReleaseCheckpoint(value) {
    const next = checkpoint(value);
    if (next.sequence !== expected.sequence || next.digest !== expected.digest) {
      throw new Error('Runtime checkpoint differs from the application-verified release history.');
    }
    const current = checkpoint(await checkpointStore.load() ?? { sequence: 0, digest: null });
    if (current.sequence === next.sequence && current.digest === next.digest) return;
    if (current.sequence !== previous.sequence || current.digest !== previous.digest
      || await checkpointStore.compareAndSwap(previous.sequence, next) !== true) {
      throw new Error('Release checkpoint changed concurrently; reverify before executing.');
    }
  }

  try {
    expected = (await verifyPackReleaseEvents(events, { pack, trustedSigners: signers, policy })).checkpoint;
  } catch (error) {
    if (error instanceof PackReleaseStateError) {
      expected = error.checkpoint;
      try { await persistReleaseCheckpoint(expected); } catch (persistenceError) {
        throw new AggregateError([error, persistenceError], 'Release rejected; its verified checkpoint could not be persisted.', { cause: error });
      }
    }
    throw error;
  }

  return { releaseEvents: events, releaseTrustedSigners: signers,
    releasePolicy: structuredClone(policy), persistReleaseCheckpoint };
}
