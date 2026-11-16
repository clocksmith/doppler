import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';

export async function hashFile(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
