import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

function filename(root, name) {
  if (typeof name !== 'string' || !name || name.split('/').some(part => !part || part === '.' || part === '..')
    || name.includes('\\') || path.isAbsolute(name)) throw new Error('Invalid storage path.');
  return path.join(root, name);
}

async function syncDirectory(directory) {
  const handle = await fs.open(directory, 'r');
  try { await handle.sync(); } finally { await handle.close(); }
}

// Private Linux application storage. One lease covers models AND document saves.
// A crashed process leaves its lease visible; never silently steal a live lease.
export async function acquireNodeStoreLease(root) {
  await fs.mkdir(root, { recursive: true, mode: 0o700 });
  const lockPath = path.join(root, '.document-search.lock');
  let lock;
  try { lock = await fs.open(lockPath, 'wx', 0o600); }
  catch (error) {
    if (error.code === 'EEXIST') throw new Error(`Storage is locked: ${lockPath}. If the recorded process crashed, confirm it has stopped before removing this file.`, { cause: error });
    throw error;
  }
  try { await lock.writeFile(JSON.stringify({ pid: process.pid, createdAtUtc: new Date().toISOString() })); await lock.sync(); }
  catch (error) { await lock.close(); await fs.unlink(lockPath); throw error; }
  let released = false;
  return async () => {
    if (released) return;
    released = true;
    await lock.close();
    await fs.unlink(lockPath);
    await syncDirectory(root);
  };
}

export async function createNodeDocumentStore(root) {
  await fs.mkdir(root, { recursive: true, mode: 0o700 });
  async function createWriteStream(name) {
    const target = filename(root, name);
    const directory = path.dirname(target);
    await fs.mkdir(directory, { recursive: true, mode: 0o700 });
    const temporary = target + '.' + randomUUID() + '.pending';
    const handle = await fs.open(temporary, 'wx', 0o600);
    let state = 'writing';
    let tail = Promise.resolve();
    let closing = null;
    let aborting = null;
    return {
      write(bytes) {
        if (state !== 'writing') return Promise.reject(new Error('Snapshot writer is closed.'));
        const owned = new Uint8Array(bytes).slice();
        tail = tail.then(() => handle.writeFile(owned));
        return tail;
      },
      close() {
        if (closing) return closing;
        if (state !== 'writing') return Promise.reject(new Error('Snapshot writer was aborted.'));
        state = 'committing';
        closing = (async () => {
          try {
            await tail;
            await handle.sync();
            await handle.close();
            await fs.rename(temporary, target);
            await syncDirectory(directory);
            state = 'committed';
          } catch (error) {
            state = 'failed';
            await handle.close();
            await fs.rm(temporary, { force: true });
            throw error;
          }
        })();
        return closing;
      },
      abort() {
        // close owns commit and reports its failure. Cleanup has already run;
        // abort must not report that same failure as a second cleanup error.
        if (closing) return closing.catch(() => {});
        if (aborting) return aborting;
        state = 'aborted';
        aborting = (async () => {
          await tail.catch(() => {});
          await handle.close();
          await fs.rm(temporary, { force: true });
        })();
        return aborting;
      },
    };
  }
  return {
    async readFile(name) {
      try { return await fs.readFile(filename(root, name)); }
      catch (error) { if (error.code === 'ENOENT') return null; throw error; }
    },
    async writeFile(name, bytes) {
      const writer = await createWriteStream(name);
      try { await writer.write(bytes); await writer.close(); }
      catch (error) { await writer.abort().catch(() => {}); throw error; }
    },
    async deleteFile(name) {
      const target = filename(root, name);
      try { await fs.unlink(target); await syncDirectory(path.dirname(target)); return true; }
      catch (error) { if (error.code === 'ENOENT') return false; throw error; }
    },
    createWriteStream,
  };
}
