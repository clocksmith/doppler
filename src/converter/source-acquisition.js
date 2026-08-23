import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const SOURCE_ACQUISITION_SCHEMA_ID = 'doppler.source-acquisition/v1';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function digest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
  return value;
}

function requireAuthor(author) {
  if (!isObject(author) || !['human', 'ai', 'tool'].includes(author.kind)
    || typeof author.actor !== 'string' || !author.actor.trim()) {
    throw new Error('Source acquisition requires attributable authorship.');
  }
}

function requireRelativeFile(value, label) {
  const file = requireString(value, label);
  if (file.startsWith('/') || file.includes('\\') || file.split('/').includes('..')) {
    throw new Error(`${label} must be a safe relative path.`);
  }
  return file;
}

function requireSha256(value, label) {
  const digestValue = requireString(value, label).replace(/^sha256:/, '');
  if (!/^[0-9a-f]{64}$/.test(digestValue)) throw new Error(`${label} must be a SHA-256 digest.`);
  return digestValue;
}

export async function createSourceAcquisitionReceipt(policy, adapters) {
  if (!isObject(policy) || policy.schema !== SOURCE_ACQUISITION_SCHEMA_ID) {
    throw new Error(`Source acquisition requires policy "${SOURCE_ACQUISITION_SCHEMA_ID}".`);
  }
  requireAuthor(policy.author);
  for (const name of ['listFiles', 'statFile', 'hashFile']) {
    if (typeof adapters?.[name] !== 'function') throw new Error(`Source acquisition requires adapter ${name}().`);
  }
  if (!Array.isArray(policy.files) || policy.files.length === 0) {
    throw new Error('Source acquisition requires a non-empty files list.');
  }
  const expected = policy.files.map((rawFile, index) => {
    if (!isObject(rawFile)) throw new Error(`files[${index}] must be an object.`);
    const size = rawFile.size;
    if (!Number.isSafeInteger(size) || size <= 0) throw new Error(`files[${index}].size must be a positive integer.`);
    return {
      path: requireRelativeFile(rawFile.path, `files[${index}].path`),
      size,
      sha256: requireSha256(rawFile.sha256, `files[${index}].sha256`),
      role: requireString(rawFile.role, `files[${index}].role`),
    };
  }).sort((left, right) => left.path.localeCompare(right.path));
  if (new Set(expected.map((file) => file.path)).size !== expected.length) {
    throw new Error('Source acquisition file paths must be unique.');
  }

  const observedPaths = (await adapters.listFiles()).map((file, index) => (
    requireRelativeFile(file, `observedFiles[${index}]`)
  )).sort();
  const expectedPaths = expected.map((file) => file.path);
  const missingFiles = expectedPaths.filter((file) => !observedPaths.includes(file));
  const unexpectedFiles = observedPaths.filter((file) => !expectedPaths.includes(file));
  if (missingFiles.length > 0 || unexpectedFiles.length > 0) {
    throw new Error(
      `Source acquisition closure failed: ${missingFiles.length} missing, ${unexpectedFiles.length} unexpected files.`
    );
  }

  const files = [];
  for (const expectedFile of expected) {
    const size = await adapters.statFile(expectedFile.path);
    if (size !== expectedFile.size) {
      throw new Error(`Source file "${expectedFile.path}" size mismatch: expected ${expectedFile.size}, got ${size}.`);
    }
    const sha256 = requireSha256(await adapters.hashFile(expectedFile.path), `observed hash for "${expectedFile.path}"`);
    if (sha256 !== expectedFile.sha256) {
      throw new Error(
        `Source file "${expectedFile.path}" digest mismatch: expected ${expectedFile.sha256}, got ${sha256}.`
      );
    }
    files.push({ ...expectedFile, observedSize: size, observedSha256: sha256, verified: true });
  }

  const core = {
    schema: 'doppler.source-acquisition-receipt/v1',
    checkpointId: requireString(policy.checkpointId, 'checkpointId'),
    repository: requireString(policy.repository, 'repository'),
    revision: requireString(policy.revision, 'revision'),
    author: structuredClone(policy.author),
    upstreamEvidence: isObject(policy.upstreamEvidence) ? structuredClone(policy.upstreamEvidence) : null,
    files,
    fileCount: files.length,
    totalBytes: files.reduce((total, file) => total + file.size, 0),
    missingFiles,
    unexpectedFiles,
    complete: true,
  };
  return Object.freeze({ ...core, receiptDigest: digest(core) });
}
