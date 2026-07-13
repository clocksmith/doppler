#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadLoRAFromManifest } from '../src/experimental/adapters/lora-loader.js';
import { exportLoRAAdapter } from '../src/experimental/training/export.js';
import {
  parseQwenPeftAdapterSafetensors,
} from '../src/experimental/training/qwen-peft-adapter-import.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MODEL_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a';
const LAYER_TYPES = Object.freeze(
  Array.from({ length: 32 }, (_, index) => (
    (index + 1) % 4 === 0 ? 'full_attention' : 'linear_attention'
  ))
);

function parseArgs(argv) {
  const args = { adapterDir: null, output: null, help: false };
  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else if (arg === '--adapter-dir') {
      args.adapterDir = argv[++index] ?? null;
    } else if (arg === '--output') {
      args.output = argv[++index] ?? null;
    } else {
      throw new Error(`Unknown argument "${arg}".`);
    }
  }
  if (!args.help && (!args.adapterDir || !args.output)) {
    throw new Error('--adapter-dir and --output are required.');
  }
  return args;
}

function printHelp() {
  console.log([
    'Usage: node tools/run-qwen-peft-adapter-import-oracle.js --adapter-dir <path> --output <path>',
    '',
    'Validates exact Qwen 3.5 9B PEFT-to-Doppler tensor names, topology,',
    'A/B transposition, and an in-memory Doppler safetensors round trip.',
  ].join('\n'));
}

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

function git(args) {
  return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
}

function tensorDigest(tensors) {
  const hash = createHash('sha256');
  for (const tensor of tensors) {
    hash.update(tensor.canonicalName);
    hash.update('\0');
    hash.update(tensor.shape.join(','));
    hash.update('\0');
    hash.update(new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength
    ));
  }
  return hash.digest('hex');
}

function summarizeShapes(tensors) {
  const groups = new Map();
  for (const tensor of tensors) {
    const key = `${tensor.projection}.lora_${tensor.kind}`;
    const shape = `[${tensor.shape.join(',')}]`;
    const group = groups.get(key) ?? new Map();
    group.set(shape, (group.get(shape) ?? 0) + 1);
    groups.set(key, group);
  }
  return Object.fromEntries(
    [...groups.entries()].sort(([left], [right]) => left.localeCompare(right)).map(
      ([key, shapes]) => [key, Object.fromEntries([...shapes.entries()].sort())]
    )
  );
}

function loadedTensors(imported, loaded) {
  return imported.tensors.map((entry) => {
    const layer = loaded.layers.get(entry.layerIndex);
    const pair = layer?.[entry.projection];
    const data = pair?.[entry.kind];
    if (!(data instanceof Float32Array)) {
      throw new Error(`Doppler round trip is missing ${entry.canonicalName}.`);
    }
    if (data.length !== entry.data.length) {
      throw new Error(`Doppler round-trip length mismatch for ${entry.canonicalName}.`);
    }
    return { ...entry, data };
  });
}

function relativeToRoot(value) {
  const absolute = path.resolve(value);
  const relative = path.relative(ROOT, absolute);
  return relative.startsWith('..') ? absolute : relative;
}

export async function runQwenPeftAdapterImportOracle(options) {
  const adapterDir = path.resolve(options.adapterDir);
  const output = path.resolve(options.output);
  const weightsPath = path.join(adapterDir, 'adapter_model.safetensors');
  const configPath = path.join(adapterDir, 'adapter_config.json');
  const [weights, configBytes, weightsStats] = await Promise.all([
    readFile(weightsPath),
    readFile(configPath),
    stat(weightsPath),
  ]);
  const adapterConfig = JSON.parse(configBytes.toString('utf8'));
  const imported = parseQwenPeftAdapterSafetensors(weights, {
    r: adapterConfig.r,
    lora_alpha: adapterConfig.lora_alpha,
    target_modules: adapterConfig.target_modules,
    layerTypes: LAYER_TYPES,
  });
  if (imported.tensorCount !== 256 || imported.pairCount !== 128) {
    throw new Error(
      `Expected the Qwen 9B V12 topology to contain 256 tensors/128 pairs; got ${imported.tensorCount}/${imported.pairCount}.`
    );
  }
  if (imported.rank !== 32 || imported.alpha !== 64) {
    throw new Error(`Expected rank 32 and alpha 64; got ${imported.rank} and ${imported.alpha}.`);
  }
  if (imported.tensors.some((tensor) => !tensor.data.every(Number.isFinite))) {
    throw new Error('Qwen PEFT adapter contains non-finite normalized values.');
  }

  const canonicalDigest = tensorDigest(imported.tensors);
  const exported = await exportLoRAAdapter({
    id: 'qwen35-9b-v12-peft-import-oracle',
    name: 'Qwen 3.5 9B V12 PEFT Import Oracle',
    description: 'In-memory round-trip oracle for Doppler-native adapter initialization.',
    baseModel: 'Qwen/Qwen3.5-9B',
    rank: imported.rank,
    alpha: imported.alpha,
    targetModules: imported.targetModules,
    tensors: imported.tensors.map((tensor) => ({
      name: tensor.canonicalName,
      shape: tensor.shape,
      dtype: 'f32',
      tensor: tensor.data,
    })),
    weightsFormat: 'safetensors',
    weightsPath: 'oracle.adapters.safetensors',
  });
  const loaded = await loadLoRAFromManifest(exported.manifest, {
    readFile: async () => exported.weights,
  });
  const roundTripDigest = tensorDigest(loadedTensors(imported, loaded));
  if (roundTripDigest !== canonicalDigest) {
    throw new Error(
      `Qwen PEFT round-trip digest mismatch: expected ${canonicalDigest}, got ${roundTripDigest}.`
    );
  }

  const sourceRevision = git(['rev-parse', 'HEAD']);
  const sourceDirty = git(['status', '--porcelain']).length > 0;
  const receipt = {
    artifactType: 'qwen35_9b_peft_adapter_import_oracle',
    schemaVersion: 1,
    passed: true,
    sourceRevision,
    sourceDirty,
    model: {
      id: 'Qwen/Qwen3.5-9B',
      revision: MODEL_REVISION,
      layerTypes: LAYER_TYPES,
      linearAttentionLayerCount: LAYER_TYPES.filter((type) => type === 'linear_attention').length,
      fullAttentionLayerCount: LAYER_TYPES.filter((type) => type === 'full_attention').length,
    },
    sourceAdapter: {
      directory: relativeToRoot(adapterDir),
      weightsPath: relativeToRoot(weightsPath),
      weightsBytes: weightsStats.size,
      weightsSha256: sha256(weights),
      configPath: relativeToRoot(configPath),
      configSha256: sha256(configBytes),
      baseModelNameOrPath: adapterConfig.base_model_name_or_path ?? null,
      peftType: adapterConfig.peft_type ?? null,
    },
    normalizedAdapter: {
      rank: imported.rank,
      alpha: imported.alpha,
      scale: imported.scale,
      targetModules: imported.targetModules,
      tensorCount: imported.tensorCount,
      pairCount: imported.pairCount,
      elementCount: imported.elementCount,
      byteCountF32: imported.elementCount * 4,
      sourceDtypes: [...new Set(imported.tensors.map((tensor) => tensor.sourceDtype))].sort(),
      canonicalTensorDigestAlgorithm: 'sha256(name_nul_shape_nul_f32_little_endian_bytes_in_canonical_order)',
      canonicalTensorDigest: canonicalDigest,
      shapes: summarizeShapes(imported.tensors),
    },
    dopplerRoundTrip: {
      passed: true,
      format: 'doppler_lora_safetensors_f32',
      tensorCount: imported.tensorCount,
      weightsBytes: exported.weights.byteLength,
      weightsSha256: exported.weightsSha256,
      manifestSha256: sha256(new TextEncoder().encode(exported.json)),
      loadedTensorDigest: roundTripDigest,
      exactDigestMatch: true,
    },
    claimBoundary: 'Exact PEFT name, topology, A/B transpose, and Doppler safetensors round-trip evidence for one completed V12 adapter. This is not the matched initial rank-32 Gamma microstep, live GPU upload at production geometry, training capability, adapter inference, or semantic WGSL evidence.',
  };
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
  return { output, receipt };
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    printHelp();
    return;
  }
  const result = await runQwenPeftAdapterImportOracle(args);
  console.log(JSON.stringify({ ok: true, output: result.output, receipt: result.receipt }, null, 2));
}

if (path.resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error?.stack || error);
    process.exitCode = 1;
  });
}
