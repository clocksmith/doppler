#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  compareFloatArrays,
  projectF16RowWiseReference,
  projectQ4KRowWiseReference,
} from './lib/q4k-projection-reference.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const REQUIRED_OPTIONS = [
  'q4-model-dir',
  'f16-model-dir',
  'q4-capture',
  'f16-capture',
  'tensor',
  'input-op',
  'output-op',
  'row',
  'artifact-dir',
  'out',
];

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help' || argument === '-h') {
      options.help = true;
      continue;
    }
    if (!argument.startsWith('--')) {
      throw new Error(`Unexpected argument "${argument}".`);
    }
    const name = argument.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) {
      throw new Error(`${argument} requires a value.`);
    }
    options[name] = value;
    index += 1;
  }
  return options;
}

function printHelp() {
  console.log([
    'Usage: node tools/q4k-projection-oracle.js [options]',
    '',
    'Required options:',
    '  --q4-model-dir <path>  Q4_K RDRR artifact directory',
    '  --f16-model-dir <path> F16 RDRR artifact directory',
    '  --q4-capture <path>    Full Q4 operator-capture receipt',
    '  --f16-capture <path>   Full F16 operator-capture receipt',
    '  --tensor <name>        Manifest tensor name',
    '  --input-op <id>        Captured projection input operator',
    '  --output-op <id>       Captured projection output operator',
    '  --row <index>          Captured activation/output row',
    '  --artifact-dir <path>  Directory for binary oracle artifacts',
    '  --out <path>           JSON receipt path',
  ].join('\n'));
}

function resolvePath(value) {
  return path.resolve(ROOT, value);
}

function sha256(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function sha256FileBytes(bytes) {
  return sha256(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
}

function portablePath(value) {
  const repositoryRelative = path.relative(ROOT, value);
  if (repositoryRelative === '' || (!repositoryRelative.startsWith(`..${path.sep}`) && repositoryRelative !== '..')) {
    return repositoryRelative || '.';
  }
  const home = os.homedir();
  if (value === home) return '~';
  if (value.startsWith(`${home}${path.sep}`)) return `~/${path.relative(home, value)}`;
  return value;
}

function repositoryCommit() {
  return execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
}

async function readJson(filename) {
  return JSON.parse(await fs.readFile(filename, 'utf8'));
}

async function readFileSlice(filename, offset, size) {
  const file = await fs.open(filename, 'r');
  try {
    const bytes = Buffer.allocUnsafe(size);
    const { bytesRead } = await file.read(bytes, 0, size, offset);
    if (bytesRead !== size) {
      throw new Error(`${filename}: expected ${size} bytes at ${offset}, read ${bytesRead}.`);
    }
    return bytes;
  } finally {
    await file.close();
  }
}

function tensorSegments(descriptor) {
  if (Array.isArray(descriptor.spans)) {
    return descriptor.spans.map((span) => ({
      shardIndex: span.shardIndex,
      offset: span.offset,
      size: span.size,
    }));
  }
  if (Number.isInteger(descriptor.shard)) {
    return [{
      shardIndex: descriptor.shard,
      offset: descriptor.offset,
      size: descriptor.size,
    }];
  }
  throw new Error('Tensor descriptor has neither spans nor a shard location.');
}

async function readTensor(modelDir, manifest, tensorName) {
  const descriptor = manifest.tensors?.[tensorName];
  if (!descriptor) {
    throw new Error(`Manifest does not contain tensor "${tensorName}".`);
  }
  const segments = tensorSegments(descriptor);
  const chunks = [];
  for (const segment of segments) {
    const shard = manifest.shards?.[segment.shardIndex];
    if (!shard?.filename) {
      throw new Error(`Tensor "${tensorName}" references missing shard ${segment.shardIndex}.`);
    }
    chunks.push(await readFileSlice(
      path.join(modelDir, shard.filename),
      segment.offset,
      segment.size
    ));
  }
  const bytes = Buffer.concat(chunks);
  if (bytes.byteLength !== descriptor.size) {
    throw new Error(`Tensor "${tensorName}" read ${bytes.byteLength} bytes, expected ${descriptor.size}.`);
  }
  return { descriptor, segments, bytes: new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength) };
}

function fullCapture(receipt, opId) {
  const matches = receipt.generation?.operatorDiagnostics?.timeline?.filter(
    (record) => record.opId === opId && record.capture?.level === 'full'
  ) ?? [];
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one full capture for "${opId}", found ${matches.length}.`);
  }
  const capture = matches[0].capture;
  if (!Array.isArray(capture.data) || !Array.isArray(capture.shape)) {
    throw new Error(`Full capture for "${opId}" is missing data or shape.`);
  }
  return capture;
}

function captureRow(capture, row) {
  const [rows, columns] = capture.shape;
  if (!Number.isInteger(rows) || !Number.isInteger(columns) || capture.data.length !== rows * columns) {
    throw new Error(`Capture shape ${JSON.stringify(capture.shape)} does not match its data.`);
  }
  if (!Number.isInteger(row) || row < 0 || row >= rows) {
    throw new Error(`Capture row ${row} is outside [0, ${rows}).`);
  }
  return Float32Array.from(capture.data.slice(row * columns, (row + 1) * columns));
}

async function writeTypedArray(filename, array) {
  const bytes = Buffer.from(array.buffer, array.byteOffset, array.byteLength);
  await fs.writeFile(filename, bytes);
  return {
    path: portablePath(filename),
    byteLength: bytes.byteLength,
    elementCount: array.length,
    sha256: sha256(bytes),
  };
}

async function writeBytes(filename, bytes) {
  const buffer = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  await fs.writeFile(filename, buffer);
  return {
    path: portablePath(filename),
    byteLength: buffer.byteLength,
    sha256: sha256(buffer),
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  for (const name of REQUIRED_OPTIONS) {
    if (options[name] === undefined) {
      throw new Error(`--${name} is required.`);
    }
  }

  const row = Number(options.row);
  if (!Number.isInteger(row) || row < 0) {
    throw new Error('--row must be a non-negative integer.');
  }

  const q4ModelDir = resolvePath(options['q4-model-dir']);
  const f16ModelDir = resolvePath(options['f16-model-dir']);
  const q4CapturePath = resolvePath(options['q4-capture']);
  const f16CapturePath = resolvePath(options['f16-capture']);
  const artifactDir = resolvePath(options['artifact-dir']);
  const outputPath = resolvePath(options.out);
  await fs.mkdir(artifactDir, { recursive: true });
  await fs.mkdir(path.dirname(outputPath), { recursive: true });

  const [q4ManifestBytes, f16ManifestBytes, q4Receipt, f16Receipt] = await Promise.all([
    fs.readFile(path.join(q4ModelDir, 'manifest.json')),
    fs.readFile(path.join(f16ModelDir, 'manifest.json')),
    readJson(q4CapturePath),
    readJson(f16CapturePath),
  ]);
  const q4Manifest = JSON.parse(q4ManifestBytes.toString('utf8'));
  const f16Manifest = JSON.parse(f16ManifestBytes.toString('utf8'));
  const [q4Tensor, f16Tensor] = await Promise.all([
    readTensor(q4ModelDir, q4Manifest, options.tensor),
    readTensor(f16ModelDir, f16Manifest, options.tensor),
  ]);

  if (q4Tensor.descriptor.dtype !== 'Q4_K_M' || q4Tensor.descriptor.layout !== 'row') {
    throw new Error(`Q4 tensor must be row-layout Q4_K_M, got ${q4Tensor.descriptor.dtype}/${q4Tensor.descriptor.layout}.`);
  }
  if (f16Tensor.descriptor.dtype !== 'F16') {
    throw new Error(`F16 tensor must have dtype F16, got ${f16Tensor.descriptor.dtype}.`);
  }
  if (JSON.stringify(q4Tensor.descriptor.shape) !== JSON.stringify(f16Tensor.descriptor.shape)) {
    throw new Error('Q4 and F16 tensor shapes do not match.');
  }

  const q4InputCapture = fullCapture(q4Receipt, options['input-op']);
  const q4OutputCapture = fullCapture(q4Receipt, options['output-op']);
  const f16InputCapture = fullCapture(f16Receipt, options['input-op']);
  const f16OutputCapture = fullCapture(f16Receipt, options['output-op']);
  const q4Activation = captureRow(q4InputCapture, row);
  const f16Activation = captureRow(f16InputCapture, row);
  const q4GpuOutput = captureRow(q4OutputCapture, row);
  const f16GpuOutput = captureRow(f16OutputCapture, row);
  if (q4Receipt.prompt !== f16Receipt.prompt) {
    throw new Error('Q4 and F16 capture prompts do not match.');
  }
  if (JSON.stringify(q4Receipt.promptTokens?.ids) !== JSON.stringify(f16Receipt.promptTokens?.ids)) {
    throw new Error('Q4 and F16 capture token IDs do not match.');
  }

  const decodedWeightsHash = crypto.createHash('sha256');
  const q4Reference = projectQ4KRowWiseReference(
    q4Tensor.bytes,
    q4Tensor.descriptor.shape,
    q4Activation,
    {
      onDecodedRow(_row, values) {
        decodedWeightsHash.update(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      },
    }
  );
  const f16ReferenceSameInput = projectF16RowWiseReference(
    f16Tensor.bytes,
    f16Tensor.descriptor.shape,
    q4Activation
  );
  const f16ReferenceOwnInput = projectF16RowWiseReference(
    f16Tensor.bytes,
    f16Tensor.descriptor.shape,
    f16Activation
  );

  const artifactFiles = {
    packedQ4: await writeBytes(path.join(artifactDir, 'packed-q4k.bin'), q4Tensor.bytes),
    decodedScales: await writeTypedArray(path.join(artifactDir, 'decoded-scales.f32'), q4Reference.decodedScales),
    decodedMinima: await writeTypedArray(path.join(artifactDir, 'decoded-minima.f32'), q4Reference.decodedMinima),
    packedScaleBits: await writeTypedArray(path.join(artifactDir, 'packed-scale-bits.u8'), q4Reference.packedScaleBits),
    packedMinBits: await writeTypedArray(path.join(artifactDir, 'packed-min-bits.u8'), q4Reference.packedMinBits),
    blockD: await writeTypedArray(path.join(artifactDir, 'block-d.f32'), q4Reference.blockD),
    blockDmin: await writeTypedArray(path.join(artifactDir, 'block-dmin.f32'), q4Reference.blockDmin),
    frozenActivation: await writeTypedArray(path.join(artifactDir, 'frozen-activation.f32'), q4Activation),
    f16Activation: await writeTypedArray(path.join(artifactDir, 'f16-activation.f32'), f16Activation),
    q4GpuOutput: await writeTypedArray(path.join(artifactDir, 'q4-gpu-output.f32'), q4GpuOutput),
    q4ReferenceOutput: await writeTypedArray(path.join(artifactDir, 'q4-reference-output.f32'), q4Reference.output),
    f16GpuOutput: await writeTypedArray(path.join(artifactDir, 'f16-gpu-output.f32'), f16GpuOutput),
    f16ReferenceSameInput: await writeTypedArray(
      path.join(artifactDir, 'f16-reference-same-input.f32'),
      f16ReferenceSameInput
    ),
    f16ReferenceOwnInput: await writeTypedArray(
      path.join(artifactDir, 'f16-reference-own-input.f32'),
      f16ReferenceOwnInput
    ),
  };

  const receipt = {
    artifactKind: 'q4k_single_projection_dequantization_oracle',
    schemaVersion: 1,
    recordedAt: new Date().toISOString(),
    repository: {
      name: 'clocksmith/doppler',
      branch: 'main',
      commit: repositoryCommit(),
    },
    host: {
      platform: process.platform,
      architecture: process.arch,
      kernel: os.release(),
      node: process.version,
      cpuModel: os.cpus()?.[0]?.model ?? null,
      logicalCpuCount: os.cpus()?.length ?? null,
      totalMemoryBytes: os.totalmem(),
      byteOrder: os.endianness(),
      metal: q4Receipt.host?.metal ?? null,
      webgpu: q4Receipt.gpu ?? null,
      privateIdentifiersRecorded: false,
    },
    invocation: {
      workingDirectory: 'repository root',
      command: ['node', 'tools/q4k-projection-oracle.js', ...process.argv.slice(2)],
    },
    projection: {
      tensorName: options.tensor,
      inputOpId: options['input-op'],
      outputOpId: options['output-op'],
      capturedRow: row,
      rowMeaning: 'last prompt token',
      shape: q4Tensor.descriptor.shape,
      q4Dtype: q4Tensor.descriptor.dtype,
      q4Layout: q4Tensor.descriptor.layout,
      f16Dtype: f16Tensor.descriptor.dtype,
    },
    prompt: {
      text: q4Receipt.prompt,
      inputTokenIds: q4Receipt.promptTokens.ids,
      identicalAcrossCaptures: true,
    },
    runtimeConfiguration: {
      q4RuntimeProfile: q4Receipt.runtimeProfile,
      f16RuntimeProfile: f16Receipt.runtimeProfile,
      q4ExecutionPlan: q4Receipt.executionPlan,
      f16ExecutionPlan: f16Receipt.executionPlan,
      q4Sampling: q4Receipt.generation?.sampling ?? null,
      f16Sampling: f16Receipt.generation?.sampling ?? null,
      q4MaximumTokens: q4Receipt.generation?.maxTokens ?? null,
      f16MaximumTokens: f16Receipt.generation?.maxTokens ?? null,
      performanceOverridesAdded: false,
    },
    artifacts: {
      q4: {
        modelId: q4Manifest.modelId,
        modelDir: portablePath(q4ModelDir),
        manifestSha256: sha256(q4ManifestBytes),
        artifactIdentity: q4Manifest.artifactIdentity,
        tensorDescriptor: q4Tensor.descriptor,
        tensorSegments: q4Tensor.segments,
        packedTensorSha256: sha256FileBytes(q4Tensor.bytes),
      },
      f16: {
        modelId: f16Manifest.modelId,
        modelDir: portablePath(f16ModelDir),
        manifestSha256: sha256(f16ManifestBytes),
        artifactIdentity: f16Manifest.artifactIdentity,
        tensorDescriptor: f16Tensor.descriptor,
        tensorSegments: f16Tensor.segments,
        packedTensorSha256: sha256FileBytes(f16Tensor.bytes),
      },
      q4Capture: {
        path: portablePath(q4CapturePath),
        sha256: sha256(await fs.readFile(q4CapturePath)),
        command: q4Receipt.invocation?.command ?? null,
      },
      f16Capture: {
        path: portablePath(f16CapturePath),
        sha256: sha256(await fs.readFile(f16CapturePath)),
        command: f16Receipt.invocation?.command ?? null,
      },
    },
    decodedQ4: {
      blockCount: q4Reference.blockD.length,
      scaleCount: q4Reference.decodedScales.length,
      minimumCount: q4Reference.decodedMinima.length,
      decodedWeightsSha256: decodedWeightsHash.digest('hex'),
      weightStats: q4Reference.decodedWeightStats,
    },
    comparisons: {
      inputQ4VsF16: compareFloatArrays(q4Activation, f16Activation),
      q4GpuVsScalarReference: compareFloatArrays(q4GpuOutput, q4Reference.output),
      f16GpuVsScalarOwnInput: compareFloatArrays(f16GpuOutput, f16ReferenceOwnInput),
      q4ScalarVsF16ScalarSameQ4Input: compareFloatArrays(q4Reference.output, f16ReferenceSameInput),
      q4GpuVsF16Gpu: compareFloatArrays(q4GpuOutput, f16GpuOutput),
      f16ScalarOwnInputVsSameQ4Input: compareFloatArrays(f16ReferenceOwnInput, f16ReferenceSameInput),
    },
    artifactFiles,
    claimBoundary: {
      trainingImprovement: 'Not evaluated or rerun.',
      baseModelInferenceCorrectness: 'This receipt tests one layer-0 projection boundary only.',
      adapterInferenceCorrectness: 'Not evaluated.',
      runtimePerformance: 'Not measured or claimed.',
    },
  };

  await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`);
  console.log(outputPath);
}

main().catch((error) => {
  console.error(error?.stack ?? String(error));
  process.exitCode = 1;
});
