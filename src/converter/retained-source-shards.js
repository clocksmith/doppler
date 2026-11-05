import { generateShardFilename } from '../formats/rdrr/manifest.js';

// Retained source shards: tensors matched by a `retain` source-tensor rule are
// not transcoded or packed. Each pinned source file that holds them becomes one
// artifact shard, appended after the packed shards, and each tensor's location
// is its exact byte span in that file. The conversion IO links the file into
// the artifact and proves its complete bytes equal the pinned SHA-256, which is
// also the shard hash the storage layer verifies.

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[RetainedSourceShards] ${message}`);
}

/** @type {typeof import('./retained-source-shards.js').partitionRetainedSourceTensors} */
export function partitionRetainedSourceTensors(tensors) {
  /** @type {typeof tensors} */
  const packed = [];
  /** @type {typeof tensors} */
  const retained = [];
  for (const tensor of tensors) (tensor.retainedSource ? retained : packed).push(tensor);
  return { packed, retained };
}

/** @type {typeof import('./retained-source-shards.js').preflightRetainedSourceShards} */
export async function preflightRetainedSourceShards(retained, io) {
  if (!retained.length) return;
  if (typeof io.retainSourceFile !== 'function') fail('this conversion surface cannot retain source files in the artifact.');
  const paths = new Set();
  for (const tensor of retained) {
    if (typeof tensor.sourcePath !== 'string' || !tensor.sourcePath) fail(`${tensor.name} lacks its retained source path.`);
    paths.add(tensor.sourcePath);
  }
  await io.preflightRetainedSources?.([...paths]);
}

/** @type {typeof import('./retained-source-shards.js').appendRetainedSourceShards} */
export async function appendRetainedSourceShards(retained, io, shards, tensorLocations, options) {
  if (retained.length === 0) return [];
  const { hashAlgorithm, manifestConfig, signal = null } = options;
  if (hashAlgorithm !== 'sha256') {
    fail(`retained shards are pinned by SHA-256 and require manifest.hashAlgorithm "sha256"; got "${hashAlgorithm}".`);
  }
  if (typeof io.retainSourceFile !== 'function') {
    fail('this conversion surface cannot link source files into the artifact.');
  }
  /** @type {Map<string, typeof retained>} */
  const byFile = new Map();
  for (const tensor of retained) {
    if (typeof tensor.sourcePath !== 'string' || !tensor.retainedSource || typeof tensor.role !== 'string'
      || typeof tensor.group !== 'string') {
      fail(`${tensor.name} lacks its retained source binding, role or group.`);
    }
    if (tensorLocations[tensor.name] !== undefined) fail(`${tensor.name} is both packed and retained.`);
    const group = byFile.get(tensor.sourcePath) ?? [];
    if (group.length > 0 && group[0].retainedSource?.sha256 !== tensor.retainedSource.sha256) {
      fail(`${tensor.sourcePath} carries two different pins.`);
    }
    group.push(tensor);
    byFile.set(tensor.sourcePath, group);
  }
  const receipt = [];
  const files = [...byFile.keys()].sort((a, b) => (byFile.get(a)?.[0].retainedSource?.file ?? '')
    .localeCompare(byFile.get(b)?.[0].retainedSource?.file ?? ''));
  for (const sourcePath of files) {
    if (signal?.aborted) throw new DOMException('Conversion cancelled', 'AbortError');
    const tensors = /** @type {typeof retained} */ (byFile.get(sourcePath));
    const { file, sha256 } = /** @type {import('./source-tensor-rules.js').RetainedSourceBinding} */ (tensors[0].retainedSource);
    const index = shards.length;
    const previous = shards[index - 1];
    const offset = previous ? previous.offset + previous.size : 0;
    const { size, hash, filename = generateShardFilename(index) } = await io.retainSourceFile(sourcePath, index, sha256);
    if (hash !== sha256) fail(`${file} hashed to ${hash}; its pin is ${sha256}.`);
    if (!filename || filename.includes('\\') || filename.startsWith('/') || filename.split('/').some(part => !part || part === '.' || part === '..') || filename.includes(':')) {
      fail(`retained shard ${index} requires a relative artifact filename; got ${JSON.stringify(filename)}.`);
    }
    shards.push({ index, filename, size, hash, offset });
    for (const tensor of tensors) {
      if (!Number.isSafeInteger(tensor.offset) || !Number.isSafeInteger(tensor.size) || tensor.offset < 0
        || tensor.size <= 0 || tensor.offset + tensor.size > size) {
        fail(`${tensor.name} span [${tensor.offset}, +${tensor.size}) lies outside ${file} (${size} bytes).`);
      }
      tensorLocations[tensor.name] = {
        shard: index,
        offset: tensor.offset,
        size: tensor.size,
        shape: tensor.shape,
        dtype: tensor.dtype,
        role: /** @type {string} */ (tensor.role),
        group: /** @type {string} */ (tensor.group),
      };
    }
    receipt.push({ shard: index, file, sha256, bytes: size, tensors: tensors.length });
  }
  manifestConfig.conversion = { ...manifestConfig.conversion, retainedSourceShards: receipt };
  return receipt;
}
