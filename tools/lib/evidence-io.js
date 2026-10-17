import fs from 'node:fs/promises';
import path from 'node:path';
import { normalizeText } from './evidence-fields.js';

/** @param {string} filePath @param {string} label */
export async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
}

/** @param {string} filePath @param {unknown} value */
export async function writeJsonAtomically(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  try {
    await fs.writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
    await fs.rename(temporaryPath, filePath);
  } finally {
    await fs.rm(temporaryPath, { force: true });
  }
}

/** @param {unknown} value */
export function isRepoRelativePath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}
