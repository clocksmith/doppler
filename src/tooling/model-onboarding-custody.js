import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { hashBytesSha256 } from '../formats/canonical-hash.js';

export async function hashOnboardingFile(filename) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filename)) hash.update(chunk);
  return `sha256:${hash.digest('hex')}`;
}

export async function retainOnboardingJson(outputDir, filename, value) {
  const text = `${JSON.stringify(value, null, 2)}\n`;
  const output = path.join(outputDir, filename);
  try { await fs.writeFile(output, text, { flag: 'wx' }); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    if (await fs.readFile(output, 'utf8') !== text) {
      throw new Error(`Retained onboarding output differs: ${filename}. Preserve it and use a new output directory.`);
    }
  }
  return { path: filename, digest: hashBytesSha256(Buffer.from(text)) };
}

export async function inventoryOnboardingFiles(directory, root = directory) {
  const files = [];
  for (const entry of (await fs.readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await inventoryOnboardingFiles(filename, root));
    else {
      if (!entry.isFile()) throw new Error(`Onboarding output must be a regular file: ${filename}`);
      files.push({ path: path.relative(root, filename), digest: await hashOnboardingFile(filename), sizeBytes: (await fs.stat(filename)).size });
    }
  }
  return files;
}
