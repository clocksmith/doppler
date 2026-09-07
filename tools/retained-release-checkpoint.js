import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

// Qualification owns an isolated copy; source application history is never rewritten.
export async function createRetainedReleaseCheckpoint(sourcePath, outputDir, semanticRoot) {
  const sourceBytes = await fs.readFile(sourcePath);
  const original = JSON.parse(sourceBytes.toString('utf8'));
  assert(original[semanticRoot], 'Retained Capsule checkpoint required.');
  const filename = path.join(outputDir, 'release-checkpoints.json');
  await fs.writeFile(filename, sourceBytes, { flag: 'wx' });
  let pending = Promise.resolve();
  async function persist(checkpoint) {
    const next = structuredClone(checkpoint);
    assert(Number.isSafeInteger(next.sequence) && next.sequence >= 0 && /^sha256:[a-f0-9]{64}$/.test(next.digest));
    const task = pending.then(async () => {
      const ledger = JSON.parse(await fs.readFile(filename, 'utf8'));
      const previous = ledger[semanticRoot];
      assert(next.sequence >= previous.sequence, 'Checkpoint rollback rejected.');
      if (next.sequence === previous.sequence) assert.equal(next.digest, previous.digest, 'Checkpoint equivocation rejected.');
      ledger[semanticRoot] = next;
      const temporary = `${filename}.pending`;
      await fs.writeFile(temporary, JSON.stringify(ledger, null, 2) + '\n', { flag: 'wx' });
      try { await fs.rename(temporary, filename); }
      catch (error) { await fs.unlink(temporary); throw error; }
    });
    pending = task.catch(() => {});
    return task;
  }
  return { filename, original, persist,
    async verifySourceUnchanged() {
      await pending;
      assert.deepEqual(await fs.readFile(sourcePath), sourceBytes, 'Source application checkpoint changed.');
    } };
}
