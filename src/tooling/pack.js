import fs from 'node:fs/promises';
import path from 'node:path';
import { freezePackV2 } from '../config/pack-v2.js';
import { validatePack, getPackIdentity } from '../config/pack.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export async function loadPack(packPath) {
  const pack = JSON.parse(await fs.readFile(path.resolve(packPath), 'utf8'));
  const validation = validatePack(pack);
  if (!validation.ok) throw new Error(`Invalid Pack at ${packPath}: ${validation.errors.join('; ')}`);
  return freezePackV2(pack);
}

export async function writePack(packPath, pack) {
  const identity = getPackIdentity(pack);
  const outputPath = path.resolve(packPath);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(stableSortObject(pack), null, 2)}\n`, 'utf8');
  return { outputPath, ...identity };
}
