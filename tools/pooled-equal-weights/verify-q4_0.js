// Untimed, byte-exact observation. Callers must read the resources actually bound
// by the engine, including source slices of fused banks, never source files.
const sha256 = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
  .map(value => value.toString(16).padStart(2, '0')).join('');
const digest = value => typeof value === 'string' && /^[a-f0-9]{64}$/u.test(value);

export function validateQ40Inventory(inventory) {
  if (inventory?.schema !== 'doppler.gguf-native-weight-inventory.v1' || inventory.completed !== true
      || !digest(inventory.sourceSha256) || !Number.isSafeInteger(inventory.chunkBytes)
      || inventory.chunkBytes < 4 || inventory.chunkBytes % 4 || !Array.isArray(inventory.tensors)
      || !inventory.tensors.length || new Set(inventory.tensors.map(tensor => tensor.name)).size !== inventory.tensors.length) {
    throw new Error('Incomplete native GGUF weight inventory.');
  }
  if (!Array.isArray(inventory.excludedTensors) || inventory.excludedTensors.some(tensor => typeof tensor?.name !== 'string'
      || !tensor.name || typeof tensor.reason !== 'string' || !tensor.reason
      || inventory.tensors.some(consumed => consumed.name === tensor.name))) {
    throw new Error('Native GGUF inventory must explicitly list every excluded source tensor with a reason.');
  }
  for (const tensor of inventory.tensors) {
    if (typeof tensor.name !== 'string' || !tensor.name || !['Q4_0', 'Q4_1', 'Q5_0', 'Q8_0', 'Q6_K', 'F32', 'F16', 'BF16'].includes(tensor.dtype)
        || !Array.isArray(tensor.shape) || !tensor.shape.length || tensor.shape.some(value => !Number.isSafeInteger(value) || value < 1)
        || !Number.isSafeInteger(tensor.bytes) || tensor.bytes < 1 || !Array.isArray(tensor.chunks)
        || tensor.chunks.length !== Math.ceil(tensor.bytes / inventory.chunkBytes)) throw new Error('Invalid native GGUF tensor inventory.');
    let covered = 0;
    for (const chunk of tensor.chunks) {
      if (chunk.offset !== covered || chunk.bytes !== Math.min(inventory.chunkBytes, tensor.bytes - covered)
          || !digest(chunk.sha256)) throw new Error(`Incomplete native GGUF chunk coverage: ${tensor.name}`);
      covered += chunk.bytes;
    }
    if (covered !== tensor.bytes) throw new Error(`Native GGUF tensor coverage differs: ${tensor.name}`);
  }
}

export async function verifyQ40Resources(inventory, resources) {
  validateQ40Inventory(inventory);
  if (!(resources instanceof Map) || resources.size !== inventory.tensors.length
      || [...resources.keys()].some(name => !inventory.tensors.some(tensor => tensor.name === name))) {
    throw new Error('Every consumed native GGUF tensor requires exactly one actual resource binding.');
  }
  const checks = [];
  for (const expected of inventory.tensors) {
    const resource = resources.get(expected.name);
    if (!resource || resource.dtype !== expected.dtype || resource.bytes !== expected.bytes
        || JSON.stringify(resource.shape) !== JSON.stringify(expected.shape)
        || !['gpu-bound', 'cpu-embedding-bound'].includes(resource.location)
        || typeof resource.readRange !== 'function') throw new Error(`Native resource representation changed: ${expected.name}`);
    for (const chunk of expected.chunks) {
      const observed = await resource.readRange(chunk.offset, chunk.bytes);
      if (!(observed instanceof ArrayBuffer) && !ArrayBuffer.isView(observed)) throw new Error('Actual resource reader must return bytes.');
      if (observed.byteLength !== chunk.bytes) throw new Error(`Short actual resource read: ${expected.name}`);
      const actual = await sha256(observed);
      if (actual !== chunk.sha256) throw new Error(`Native GGUF materialization changed: ${expected.name} at ${chunk.offset}`);
      checks.push({ name: expected.name, offset: chunk.offset, bytes: chunk.bytes, sha256: actual, location: resource.location });
    }
  }
  return { schema: 'doppler.gguf-native-materialization.v1', pass: true, sourceSha256: inventory.sourceSha256,
    excludedTensors: inventory.excludedTensors, tensorCount: inventory.tensors.length, byteCount: inventory.tensors.reduce((sum, tensor) => sum + tensor.bytes, 0), checks };
}

export function verifyQ40Numerical(contract, prompts, doppler, pooled) {
  if (!Array.isArray(prompts) || !prompts.length || new Set(prompts.map(prompt => prompt.id)).size !== prompts.length
      || doppler?.length !== prompts.length || pooled?.length !== prompts.length) throw new Error('Incomplete native numerical prompt coverage.');
  const checks = [];
  for (const prompt of prompts) {
    const a = doppler.find(row => row.id === prompt.id), b = pooled.find(row => row.id === prompt.id);
    if (!a || !b || JSON.stringify(a.promptIds) !== JSON.stringify(prompt.promptIds)
        || JSON.stringify(a.promptIds) !== JSON.stringify(b.promptIds)
        || !Array.isArray(a.tokenIds) || a.tokenIds.length !== contract.shared.maxNewTokens
        || JSON.stringify(a.tokenIds) !== JSON.stringify(b.tokenIds)) throw new Error(`Native token parity failed: ${prompt.id}`);
    for (const row of [a, b]) {
      if (row.logits?.length !== contract.gates.logitPositions.length
          || new Set(row.logits.map(logits => logits.step)).size !== row.logits.length) throw new Error('Incomplete native logit position coverage.');
    }
    for (const step of contract.gates.logitPositions) {
      const x = a.logits.find(row => row.step === step)?.values, y = b.logits.find(row => row.step === step)?.values;
      if (!Array.isArray(x) || !Array.isArray(y) || x.length !== contract.vocabSize || y.length !== contract.vocabSize) throw new Error('Complete native vocabulary logits required.');
      let maxAbs = 0;
      for (let index = 0; index < x.length; index++) {
        if (!Number.isFinite(x[index]) || !Number.isFinite(y[index])) throw new Error('Nonfinite native logits.');
        maxAbs = Math.max(maxAbs, Math.abs(x[index] - y[index]));
      }
      if (maxAbs > contract.gates.logitAbsoluteTolerance) throw new Error(`Native logit parity failed: ${prompt.id}/${step}: ${maxAbs}`);
      checks.push({ prompt: prompt.id, step, maxAbs });
    }
  }
  return { schema: 'doppler.gguf-native-numerical.v1', pass: true, checks };
}
