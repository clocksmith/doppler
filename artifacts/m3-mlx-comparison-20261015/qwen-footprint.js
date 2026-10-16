import fs from 'node:fs/promises';
import { resolveTensorTargetQuant, shouldQuantize, shouldSkipModuleQuantization } from '../../src/converter/tensor-transform.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
installNodeFileFetchShim();
const root = new URL('./', import.meta.url);
const inventory = JSON.parse(await fs.readFile(new URL('qwen27b-source-headers.json', root)));
const config = JSON.parse(await fs.readFile('src/config/conversion/qwen3/qwen3.8-27b-text-q4k-ehaf16.json'));
const groups = {}, tensors = [];
for (const shard of inventory.shards) for (const [name, tensor] of Object.entries(shard.tensors)) {
  // Exact text-only subset of this source; vision and MTP are excluded.
  if (!name.startsWith('model.language_model.') && name !== 'lm_head.weight') continue;
  const item = {name, ...tensor};
  const target = resolveTensorTargetQuant(item, config.quantization.weights, config.quantization);
  const packed = target === 'q4k' && shouldQuantize(name, tensor.shape, {
    quantizeEmbeddings: false, modulesToNotConvert: config.quantization.modulesToNotConvert,
  });
  const elements = tensor.shape.reduce((a, b) => a * b, 1);
  const denseBytes = elements * (tensor.dtype === 'F32' && target === 'q4k' ? 4 : 2);
  if (packed && tensor.shape.length !== 2) throw new Error(`Unexpected packed rank for ${name}`);
  const bytes = packed ? tensor.shape[0] * Math.ceil(tensor.shape[1] / 256) * 144 : denseBytes;
  const group = packed ? 'packedQ4K' : shouldSkipModuleQuantization(name, config.quantization.modulesToNotConvert)
    ? 'excludedLinearAttention' : name.includes('embed_tokens') ? 'embeddingF16'
    : name === 'lm_head.weight' ? 'headF16' : 'otherDense';
  groups[group] ??= {tensors: 0, bytes: 0, elements: 0};
  groups[group].tensors++; groups[group].bytes += bytes; groups[group].elements += elements;
  tensors.push({name, shape: tensor.shape, sourceDtype: tensor.dtype, estimatedStorageBytes: bytes, group});
}
const totalBytes = Object.values(groups).reduce((sum, group) => sum + group.bytes, 0);
const result = {
  source: inventory.source,
  scope: 'Header-derived estimate for existing text-only Q4K conversion policy. Not measured residency or fit qualification.',
  formula: 'Row Q4_K uses ceil(columns / 256) * 144 bytes per row; F16 exclusions use 2 bytes/element; skipped F32 tensors retain 4.',
  omitted: ['GPU row padding and alignment', 'KV and recurrent state', 'activations and temporary buffers', 'CPU copies, conversion workspace, OS and other applications'],
  totalBytes, totalGiB: totalBytes / 2 ** 30,
  hostPhysicalBytes: 24 * 2 ** 30, physicalRemainderBeforeOmittedCostsBytes: 24 * 2 ** 30 - totalBytes,
  groups, tensors,
};
await fs.writeFile(new URL('qwen27b-footprint.json', root), JSON.stringify(result, null, 2));
console.log(JSON.stringify({totalGiB: result.totalGiB, groups}, null, 2));
