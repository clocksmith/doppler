import fs from 'node:fs/promises';
import path from 'node:path';
import { freezeCapsuleV2 } from '../config/capsule-v2.js';
import { validateCapsule, getCapsuleIdentity } from '../config/capsule.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export async function loadCapsule(capsulePath, options = {}) {
  const capsule = JSON.parse(await fs.readFile(path.resolve(capsulePath), { encoding: 'utf8', signal: options.signal ?? undefined }));
  const validation = validateCapsule(capsule);
  if (!validation.ok) throw new Error(`Invalid Capsule at ${capsulePath}: ${validation.errors.join('; ')}`);
  return freezeCapsuleV2(capsule);
}

export async function writeCapsule(capsulePath, capsule) {
  const identity = getCapsuleIdentity(capsule);
  const outputPath = path.resolve(capsulePath);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(stableSortObject(capsule), null, 2)}\n`, 'utf8');
  return { outputPath, ...identity };
}
