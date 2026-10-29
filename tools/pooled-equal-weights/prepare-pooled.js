import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';

// A separately labeled control, never an invisible change to upstream Pooled.
export async function preparePooledRounding(snapshotDir, policy) {
  if (policy == null) return [];
  if (policy.mode !== 'explicit-nearest-even') throw new Error('Unsupported Pooled rounding preparation');
  const changes = [];
  for (const name of ['qwen35.js', 'layer_fuse.js']) {
    const relative = `engine/wgsl/${name}`, file = path.join(snapshotDir, relative);
    const original = await fs.readFile(file, 'utf8');
    const sha = text => createHash('sha256').update(text).digest('hex');
    if (sha(original) !== policy.originalSha256?.[relative]) throw new Error(`Pooled rounding source identity mismatch: ${relative}`);
    if ([...original.matchAll(/\bpack2x16float\(/g)].length !== 2) throw new Error(`Pooled KV packing sites changed: ${relative}`);
    let corrected = original.replace(/\bpack2x16float\(/g, 'pack2x16_rne(');
    if (name === 'qwen35.js') {
      const anchor = 'export const WGSL2 = /* wgsl */ `';
      if (corrected.split(anchor).length !== 2) throw new Error('Pooled WGSL module boundary changed');
      corrected = 'import { F16_WGSL } from "./f16.js";\n' + corrected.replace(anchor, anchor + '\n${F16_WGSL}\n');
    }
    await fs.writeFile(file, corrected);
    changes.push({ path: relative, originalSha256: sha(original), preparedSha256: sha(corrected),
      change: 'Use existing upstream nearest-even helper for the two KV packing sites; no weight, geometry, or scheduling changes.' });
  }
  return changes;
}
