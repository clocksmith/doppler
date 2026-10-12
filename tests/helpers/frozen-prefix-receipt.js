import { createWriteStream } from 'node:fs';
import { rename, rm } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';

// Captured tensor scalars are bounded; the aggregate timeline can exceed V8's
// single-string limit. Never stringify that entire timeline or receipt at once.
export function* frozenPrefixJsonChunks(value) {
  if (Array.isArray(value)) {
    yield '[';
    for (let index = 0; index < value.length; index++) {
      if (index > 0) yield ',';
      if (typeof value[index] === 'number') {
        const start = index;
        while (index + 1 < value.length && index - start < 1023 && typeof value[index + 1] === 'number') index++;
        yield JSON.stringify(value.slice(start, index + 1)).slice(1, -1);
      } else yield* frozenPrefixJsonChunks(value[index] ?? null);
    }
    yield ']';
  } else if (value !== null && typeof value === 'object') {
    yield '{';
    let first = true;
    for (const [key, field] of Object.entries(value)) {
      if (field === undefined) continue;
      if (!first) yield ',';
      first = false;
      yield JSON.stringify(key) + ':';
      yield* frozenPrefixJsonChunks(field);
    }
    yield '}';
  } else yield JSON.stringify(value);
}

export async function writeFrozenPrefixReceipt(destination, receipt) {
  const partial = destination + '.partial';
  await pipeline(frozenPrefixJsonChunks(receipt), createWriteStream(partial));
  await rename(partial, destination);
}

export async function finishFrozenPrefixReplay({ destination, receipt, browser, server, directory,
  writeReceipt = writeFrozenPrefixReceipt }) {
  const errors = [];
  const steps = [
    () => writeReceipt(destination, receipt),
    () => browser?.close(),
    async () => {
      if (server) {
        server.closeAllConnections();
        await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      }
    },
    () => rm(directory, { recursive: true, force: true }),
  ];
  for (const step of steps) {
    try { await step(); } catch (error) { errors.push(error); }
  }
  if (errors.length === 1) throw errors[0];
  if (errors.length > 1) throw new AggregateError(errors, 'Prefix receipt or owned-resource cleanup failed');
}
