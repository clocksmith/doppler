import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { parseGGUFHeader } from '../../src/formats/gguf/types.js';
import { parseGGUFModel } from '../../src/converter/parsers/gguf.js';
import { HEADER_READ_SIZE } from '../../src/config/schema/index.js';
import { validateQ40Inventory } from './verify-q4_0.js';

export async function prepareQ40Inventory(file, chunkBytes) {
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes < 4 || chunkBytes % 4) throw new Error('Declare a positive aligned observation chunkBytes.');
  const handle = await fs.open(file, 'r');
  try {
    const { size } = await handle.stat();
    const readRange = async (offset, bytes) => {
      const output = Buffer.alloc(bytes), read = await handle.read(output, 0, bytes, offset);
      if (read.bytesRead !== bytes) throw new Error('Incomplete GGUF source read.');
      return output;
    };
    const header = await readRange(0, Math.min(size, HEADER_READ_SIZE));
    const parsed = await parseGGUFModel({ file: { size, readRange }, normalizeTensorSource: value => value,
      parseGGUFHeaderFromSource: async () => parseGGUFHeader(header.buffer.slice(header.byteOffset, header.byteOffset + header.byteLength)) });
    const hash = createHash('sha256');
    for await (const bytes of createReadStream(file)) hash.update(bytes);
    const inventory = { schema: 'doppler.gguf-native-weight-inventory.v1', completed: true,
      sourceSha256: hash.digest('hex'), chunkBytes, excludedTensors: parsed.excludedTensors, tensors: [] };
    for (const tensor of parsed.tensors) {
      const chunks = [];
      for (let offset = 0; offset < tensor.size; offset += chunkBytes) {
        const bytes = Math.min(chunkBytes, tensor.size - offset);
        const observed = await readRange(tensor.offset + offset, bytes);
        chunks.push({ offset, bytes, sha256: createHash('sha256').update(observed).digest('hex') });
      }
      inventory.tensors.push({ name: tensor.name, shape: tensor.shape, dtype: tensor.dtype, bytes: tensor.size, chunks });
    }
    validateQ40Inventory(inventory);
    return inventory;
  } finally { await handle.close(); }
}
if (process.argv[1] && import.meta.url === new URL(process.argv[1], 'file:').href) {
  const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
  const inventory = await prepareQ40Inventory(config.ggufPath, config.chunkBytes);
  await fs.writeFile(config.inventoryPath, JSON.stringify(inventory, null, 2) + '\n', { flag: 'wx' });
}
