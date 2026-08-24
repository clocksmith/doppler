import { initDevice, getKernelCapabilities, getDevice } from '../../../gpu/device.js';
import { setPlatformsBaseUrl } from '../../../config/platforms/loader.js';
import { setRegistryUrl } from '../../../config/kernels/registry.js';
import { createTrainingConfig } from '../../../config/training-defaults.js';
import {
  runMatmul,
  runResidualAdd,
} from '../../../gpu/kernels/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../../memory/buffer-pool.js';
import { OpType } from '../autograd.js';
import { AdamOptimizer } from '../optimizer.js';
import { TrainingRunner } from '../runner.js';
import { trainStep } from '../trainer.js';
import { crossEntropyLoss } from '../loss.js';
import { clipGradients } from '../clip.js';
import { exportLoRAAdapter } from '../export.js';
import { sha256Hex } from '../../../formats/sha256.js';
import { computeSampleStats } from '../../../debug/stats.js';
import { parseJsonl } from '../datasets/jsonl.js';
import {
  buildDistillCandidatePrompt,
  buildDistillPrompt,
  encodeDistillRow,
  normalizeDistillDatasetPath,
  normalizeOptionalString,
  resolveDistillDataScope,
  summarizeDirectionCounts,
} from '../distillation/suite-data.js';
import { createDistillStudentRuntimeModelFixture } from '../distillation/student-fixture.js';
import { initializeInference } from '../../../inference/test-harness.js';
import { createPipeline } from '../../../inference/pipelines/text.js';
import { parseManifest } from '../../../formats/rdrr/index.js';
import { openModelStore, loadManifestFromStore } from '../../../storage/shard-manager.js';

export const DISTILL_ADAPTER_TOP_K = 64;

export const DISTILL_LOGIT_FALLBACK = -80;

export const DISTILL_STUDENT_GRAPH_PROJECTION = 'projection_head';

export const DISTILL_STUDENT_GRAPH_FULL = 'transformer_full';

export function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tensor');
}

export function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tokens');
  uploadData(buffer, data);
  // Token tensors are wrapped as f32 by contract; kernels read the underlying u32 bytes.
  return createTensor(buffer, 'f32', shape, label || 'train_tokens');
}

export function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
}

export function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

export function toFiniteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function clampDistillTopK(value) {
  const parsed = Math.floor(toFiniteNumber(value, DISTILL_ADAPTER_TOP_K));
  return Math.max(2, Math.min(256, parsed));
}

export function normalizeDistillStudentGraphMode(value) {
  const normalized = normalizeOptionalString(value);
  if (!normalized) return DISTILL_STUDENT_GRAPH_FULL;
  const compact = normalized.toLowerCase().replace(/[-\s]/g, '_');
  if (compact === 'projection_head' || compact === 'projection') {
    return DISTILL_STUDENT_GRAPH_PROJECTION;
  }
  return DISTILL_STUDENT_GRAPH_FULL;
}

export function toFloat32Array(values, label = 'values') {
  if (values instanceof Float32Array) return values;
  if (ArrayBuffer.isView(values)) {
    return new Float32Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  }
  if (Array.isArray(values)) {
    return new Float32Array(values);
  }
  throw new Error(`Distill ${label} must be an array-like float buffer.`);
}

export function selectTopKIndices(logits, topK) {
  const k = Math.max(1, Math.floor(topK));
  const indices = new Int32Array(k);
  const values = new Float32Array(k);
  indices.fill(-1);
  values.fill(-Infinity);

  for (let i = 0; i < logits.length; i += 1) {
    const value = Number.isFinite(logits[i]) ? logits[i] : DISTILL_LOGIT_FALLBACK;
    if (value <= values[k - 1]) continue;
    let insert = k - 1;
    while (insert > 0 && value > values[insert - 1]) {
      values[insert] = values[insert - 1];
      indices[insert] = indices[insert - 1];
      insert -= 1;
    }
    values[insert] = value;
    indices[insert] = i;
  }

  for (let i = 0; i < k; i += 1) {
    if (indices[i] >= 0) continue;
    indices[i] = i < logits.length ? i : -1;
  }
  return indices;
}

export function gatherLogitsByIndices(logits, indices, fallback = DISTILL_LOGIT_FALLBACK) {
  const gathered = new Float32Array(indices.length);
  for (let i = 0; i < indices.length; i += 1) {
    const tokenIndex = indices[i];
    if (tokenIndex >= 0 && tokenIndex < logits.length) {
      const value = logits[tokenIndex];
      gathered[i] = Number.isFinite(value) ? value : fallback;
      continue;
    }
    gathered[i] = fallback;
  }
  return gathered;
}

export function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const value = Number.isFinite(values[i]) ? values[i] : Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}

export function softmax(values, temperature = 1) {
  const t = Math.max(1e-4, toFiniteNumber(temperature, 1));
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    const candidate = values[i] / t;
    if (candidate > max) max = candidate;
  }
  const exps = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = Math.exp((values[i] / t) - max);
    exps[i] = value;
    sum += value;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    const uniform = 1 / Math.max(1, values.length);
    exps.fill(uniform);
    return exps;
  }
  for (let i = 0; i < exps.length; i += 1) {
    exps[i] /= sum;
  }
  return exps;
}

export function disposePrefillSnapshot(result) {
  const cache = result?.cache;
  if (cache && typeof cache.clear === 'function') {
    cache.clear();
  }
}

export function buildShuffledIndices(length, seed = 1337) {
  const indices = Array.from({ length }, (_, idx) => idx);
  let state = (Number(seed) >>> 0) || 0x6d2b79f5;
  for (let i = indices.length - 1; i > 0; i -= 1) {
    state = ((state * 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1);
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  return indices;
}

export function normalizeDistillStage(value) {
  const stage = String(value || '').trim();
  return stage === 'stage_b' ? 'stage_b' : 'stage_a';
}

export async function computeTeacherPromptDistillFeatures(sample, prompt, runtime) {
  const teacherResult = await runtime.teacherPipeline.prefillWithLogits(prompt, {
    useChatTemplate: false,
  });
  try {
    const teacherLogits = toFloat32Array(teacherResult?.logits, 'teacher logits');
    const topTokenIndices = selectTopKIndices(teacherLogits, runtime.topK);
    const teacherTopLogits = gatherLogitsByIndices(teacherLogits, topTokenIndices, DISTILL_LOGIT_FALLBACK);
    const teacherTopProbs = softmax(teacherTopLogits, runtime.temperature);
    const targetClass = argmax(teacherTopLogits);
    return {
      source: sample.source,
      direction: sample.direction,
      targetClass,
      topTokenIndices: Array.from(topTokenIndices),
      teacherTopLogits,
      teacherTopProbs,
    };
  } finally {
    disposePrefillSnapshot(teacherResult);
    runtime.teacherPipeline.reset();
  }
}

export function createDistillTensorDataset(samples, options = {}) {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new Error('Distill dataset has no usable rows.');
  }
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  if (!distillRuntime?.teacherPipeline) {
    throw new Error('Distill dataset requires teacherPipeline.');
  }
  const batchSize = Math.max(1, Math.floor(Number(options.batchSize) || 1));
  const shuffle = options.shuffle === true;
  const seed = Number.isInteger(options.seed) ? options.seed : 1337;
  const stage = normalizeDistillStage(distillRuntime.stage);
  const topK = clampDistillTopK(distillRuntime.topK);

  return {
    async *batches() {
      const order = shuffle
        ? buildShuffledIndices(samples.length, seed)
        : Array.from({ length: samples.length }, (_, idx) => idx);
      let inputTensor = null;
      let targetTensor = null;
      let tensorBatchSize = 0;
      try {
        for (let offset = 0; offset < order.length; offset += batchSize) {
          const batchIndices = order.slice(offset, offset + batchSize);
          const features = new Float32Array(batchIndices.length * topK);
          const targets = new Uint32Array(batchIndices.length);
          const teacherTopProbs = [];
          const teacherTopTokenIndices = [];
          const teacherTopLogits = [];
          const teacherTargetIndices = [];
          const teacherTargetTokenIds = [];
          const prompts = [];
          const tripletPositivePrompts = [];
          const tripletNegativePrompts = [];
          const tripletMask = [];
          const directionCounts = {};

          for (let i = 0; i < batchIndices.length; i += 1) {
            const sample = samples[batchIndices[i]];
            const prompt = buildDistillPrompt(sample);
            const baseDistill = await computeTeacherPromptDistillFeatures(sample, prompt, {
              ...distillRuntime,
              topK,
            });

            const baseOffset = i * topK;
            features.set(baseDistill.teacherTopLogits, baseOffset);
            const targetClass = baseDistill.targetClass;
            const targetToken = Number.isInteger(baseDistill.topTokenIndices?.[targetClass])
              ? baseDistill.topTokenIndices[targetClass]
              : targetClass;
            const targetTokenMode = distillRuntime.targetTokenMode === 'teacher_top_token';
            targets[i] = targetTokenMode ? targetToken : targetClass;
            teacherTargetIndices.push(targetClass);
            teacherTargetTokenIds.push(targetToken);
            teacherTopProbs.push(baseDistill.teacherTopProbs);
            teacherTopTokenIndices.push(baseDistill.topTokenIndices);
            teacherTopLogits.push(baseDistill.teacherTopLogits);
            prompts.push(prompt);

            if (stage === 'stage_b') {
              const posPrompt = buildDistillCandidatePrompt(sample, sample.targetPos);
              const negPrompt = sample.targetNeg
                ? buildDistillCandidatePrompt(sample, sample.targetNeg)
                : null;
              tripletPositivePrompts.push(posPrompt);
              tripletNegativePrompts.push(negPrompt || posPrompt);
              tripletMask.push(Boolean(negPrompt));
            }

            directionCounts[sample.direction] = (directionCounts[sample.direction] || 0) + 1;
          }

          if (!inputTensor || !targetTensor || tensorBatchSize !== batchIndices.length) {
            releaseTensor(inputTensor);
            releaseTensor(targetTensor);
            inputTensor = makeTensorFromFloat32(
              features,
              [batchIndices.length, topK],
              'distill_jsonl_input'
            );
            targetTensor = makeTensorFromUint32(
              targets,
              [batchIndices.length],
              'distill_jsonl_targets'
            );
            tensorBatchSize = batchIndices.length;
          } else {
            uploadData(inputTensor.buffer, features);
            uploadData(targetTensor.buffer, targets);
          }
          yield {
            input: inputTensor,
            targets: targetTensor,
            distill: {
              prompts,
              tripletPositivePrompts,
              tripletNegativePrompts,
              tripletMask,
              teacherTopProbs,
              teacherTopTokenIndices,
              teacherTopLogits,
              teacherTargetIndices,
              teacherTargetTokenIds,
              targetTokenMode: distillRuntime.targetTokenMode || 'topk_class',
              batchSampleCount: batchIndices.length,
              directionCounts,
              distillStage: stage,
              temperature: toFiniteNumber(distillRuntime.temperature, 1),
              alphaKd: toFiniteNumber(distillRuntime.alphaKd, 1),
              alphaCe: toFiniteNumber(distillRuntime.alphaCe, 0),
              tripletMargin: Math.max(0, toFiniteNumber(distillRuntime.tripletMargin, 0.2)),
              teacherModelId: distillRuntime.teacherModelId || null,
              studentModelId: distillRuntime.studentModelId || null,
            },
          };
        }
      } finally {
        releaseTensor(inputTensor);
        releaseTensor(targetTensor);
      }
    },
  };
}

export async function loadDistillDatasetFromJsonl(datasetPath, scopeOptions = null) {
  const normalizedPath = normalizeDistillDatasetPath(datasetPath);
  if (!normalizedPath) return null;
  if (!isNodeRuntime()) {
    throw new Error('distillDatasetPath currently requires Node runtime.');
  }
  const normalizedScope = (
    scopeOptions && typeof scopeOptions === 'object'
      ? scopeOptions
      : resolveDistillDataScope()
  );

  const [{ readFile }, { resolve, dirname, isAbsolute, join, sep }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
  ]);

  const isShardManifest = (candidate) => {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) return false;
    if (!Array.isArray(candidate.shards) || candidate.shards.length === 0) return false;
    return candidate.shards.every((entry) => {
      if (typeof entry === 'string' && entry.trim()) return true;
      if (entry && typeof entry === 'object' && typeof entry.path === 'string' && entry.path.trim()) return true;
      return false;
    });
  };
  const resolveShardPath = (entry, manifestDir) => {
    const rawPath = typeof entry === 'string' ? entry : entry.path;
    const normalized = String(rawPath || '').trim();
    if (!normalized) return null;
    if (isAbsolute(normalized)) return normalized;
    if (normalized.startsWith(`.${sep}`) || normalized.startsWith(`..${sep}`)) {
      return resolve(manifestDir, normalized);
    }
    const projectsPrefix = `projects${sep}`;
    if (normalized.startsWith(projectsPrefix)) {
      const marker = `${sep}projects${sep}`;
      const markerIndex = manifestDir.lastIndexOf(marker);
      if (markerIndex >= 0) {
        const workspaceRoot = manifestDir.slice(0, markerIndex);
        return join(workspaceRoot, normalized);
      }
    }
    return join(manifestDir, normalized);
  };
  const loadEncodedRows = (rawRows, contextLabel) => {
    const encodedRows = [];
    for (let i = 0; i < rawRows.length; i += 1) {
      let encoded = null;
      try {
        encoded = encodeDistillRow(rawRows[i], i, normalizedScope);
      } catch (error) {
        const message = error?.message ? String(error.message) : String(error);
        throw new Error(`${contextLabel}: row ${i + 1}: ${message}`);
      }
      if (encoded) encodedRows.push(encoded);
    }
    return encodedRows;
  };

  const absolutePath = resolve(normalizedPath);
  let raw;
  try {
    raw = await readFile(absolutePath, 'utf8');
  } catch (error) {
    const message = error?.message ? String(error.message) : String(error);
    throw new Error(`Failed to read distillDatasetPath "${absolutePath}": ${message}`);
  }

  let parsedJson = null;
  try {
    parsedJson = JSON.parse(raw);
  } catch {
    parsedJson = null;
  }
  if (isShardManifest(parsedJson)) {
    const manifestDir = dirname(absolutePath);
    const shardPaths = parsedJson.shards
      .map((entry) => resolveShardPath(entry, manifestDir))
      .filter(Boolean);
    if (shardPaths.length === 0) {
      throw new Error(`Distill shard manifest "${absolutePath}" has no valid shard paths.`);
    }
    let rowCount = 0;
    let sampleCount = 0;
    const directionCounts = {};
    for (const shardPath of shardPaths) {
      const shardRaw = await readFile(shardPath, 'utf8');
      const shardRows = parseJsonl(shardRaw);
      const encodedRows = loadEncodedRows(shardRows, `distill shard "${shardPath}"`);
      rowCount += shardRows.length;
      sampleCount += encodedRows.length;
      const shardDirections = summarizeDirectionCounts(encodedRows);
      for (const [direction, count] of Object.entries(shardDirections)) {
        directionCounts[direction] = (directionCounts[direction] || 0) + count;
      }
    }
    if (sampleCount <= 0) {
      throw new Error(`Distill shard manifest "${absolutePath}" has no usable rows across shards.`);
    }
    return {
      absolutePath,
      rowCount,
      sampleCount,
      directionCounts,
      dataScope: {
        sourceLangs: normalizedScope.sourceLangs || null,
        targetLangs: normalizedScope.targetLangs || null,
        pairAllowlist: normalizedScope.pairAllowlist || null,
        strictPairContract: normalizedScope.strictPairContract === true,
      },
      shardCount: shardPaths.length,
      shardPaths,
      createDataset(runOptions = {}) {
        const shardSeedBase = Number.isInteger(runOptions.seed) ? runOptions.seed : 1337;
        return {
          async *batches() {
            for (let shardIndex = 0; shardIndex < shardPaths.length; shardIndex += 1) {
              const shardPath = shardPaths[shardIndex];
              const shardRaw = await readFile(shardPath, 'utf8');
              const shardRows = parseJsonl(shardRaw);
              const encodedRows = loadEncodedRows(shardRows, `distill shard "${shardPath}"`);
              if (encodedRows.length === 0) continue;
              const shardDataset = createDistillTensorDataset(encodedRows, {
                ...runOptions,
                seed: shardSeedBase + shardIndex,
              });
              for await (const batch of shardDataset.batches()) {
                if (batch?.distill && typeof batch.distill === 'object') {
                  batch.distill.datasetShardIndex = shardIndex + 1;
                  batch.distill.datasetShardCount = shardPaths.length;
                  batch.distill.datasetShardPath = shardPath;
                }
                yield batch;
              }
            }
          },
        };
      },
    };
  }

  const rows = parseJsonl(raw);
  const encodedRows = loadEncodedRows(rows, `distill dataset "${absolutePath}"`);
  if (encodedRows.length === 0) {
    throw new Error(`Distill dataset "${absolutePath}" has no usable rows.`);
  }

  return {
    absolutePath,
    rowCount: rows.length,
    sampleCount: encodedRows.length,
    directionCounts: summarizeDirectionCounts(encodedRows),
    dataScope: {
      sourceLangs: normalizedScope.sourceLangs || null,
      targetLangs: normalizedScope.targetLangs || null,
      pairAllowlist: normalizedScope.pairAllowlist || null,
      strictPairContract: normalizedScope.strictPairContract === true,
    },
    createDataset(runOptions = {}) {
      return createDistillTensorDataset(encodedRows, runOptions);
    },
  };
}

export function looksLikeUrl(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(String(value || '').trim());
}

export function looksLikeFilesystemPath(value) {
  const normalized = String(value || '').trim();
  return normalized.startsWith('/') || normalized.startsWith('./') || normalized.startsWith('../');
}

export async function resolveNodeModelUrlFromRef(modelRef) {
  if (!isNodeRuntime()) return null;
  const [{ access, constants }, { resolve, join }, { pathToFileURL }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
    import('node:url'),
  ]);

  const normalized = String(modelRef || '').trim();
  if (!normalized) return null;
  const candidates = [
    normalized,
    join('models', 'local', normalized),
    join('models', 'curated', normalized),
  ];
  for (const candidate of candidates) {
    const absolutePath = resolve(candidate);
    const manifestPath = join(absolutePath, 'manifest.json');
    try {
      await access(manifestPath, constants.R_OK);
      return pathToFileURL(absolutePath).href;
    } catch {
      // Try next candidate.
    }
  }
  return null;
}

export async function initializeInferenceFromStore(modelId) {
  await openModelStore(modelId);
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error(`Manifest not found in store for model "${modelId}".`);
  }
  const manifest = parseManifest(manifestText);
  const pipeline = await createPipeline(manifest, {
    gpu: { device: getDevice() },
  });
  return { pipeline, manifest };
}

export async function loadDistillModelHandle(modelRef, role, loadOptions = {}) {
  const normalizedRef = normalizeOptionalString(modelRef);
  if (!normalizedRef) {
    throw new Error(`Distill ${role} model reference is required.`);
  }

  const loadFromUrl = async (url) => {
    const initialized = await initializeInference(url, {
      log: () => {},
      onProgress: () => {},
      runtime: loadOptions.runtime || undefined,
    });
    return {
      modelRef: normalizedRef,
      modelUrl: url,
      manifest: initialized.manifest,
      pipeline: initialized.pipeline,
    };
  };

  if (looksLikeUrl(normalizedRef)) {
    return loadFromUrl(normalizedRef);
  }

  if (isNodeRuntime()) {
    const localUrl = await resolveNodeModelUrlFromRef(normalizedRef);
    if (localUrl) {
      return loadFromUrl(localUrl);
    }
  }

  if (looksLikeFilesystemPath(normalizedRef) && isNodeRuntime()) {
    const [{ resolve }, { pathToFileURL }] = await Promise.all([
      import('node:path'),
      import('node:url'),
    ]);
    return loadFromUrl(pathToFileURL(resolve(normalizedRef)).href);
  }

  const { pipeline, manifest } = await initializeInferenceFromStore(normalizedRef);
  return {
    modelRef: normalizedRef,
    modelUrl: null,
    manifest,
    pipeline,
  };
}

export function resolveDistillModelRefs(options = {}, trainingConfig = null) {
  const distillConfig = trainingConfig?.distill || {};
  return {
    teacherModelRef: normalizeOptionalString(options.teacherModelId ?? distillConfig.teacherModelId),
    studentModelRef: normalizeOptionalString(options.studentModelId ?? distillConfig.studentModelId),
  };
}

export async function createDistillRuntimeContext(options = {}, trainingConfig = null) {
  const { teacherModelRef, studentModelRef } = resolveDistillModelRefs(options, trainingConfig);
  if (!teacherModelRef || !studentModelRef) {
    throw new Error('Distill stage requires teacherModelId and studentModelId.');
  }

  const distillConfig = trainingConfig?.distill || {};
  const studentGraphMode = normalizeDistillStudentGraphMode(
    options.studentGraphMode
    ?? distillConfig.studentGraphMode
  );
  const teacher = await loadDistillModelHandle(teacherModelRef, 'teacher');
  const studentRuntime = studentGraphMode === DISTILL_STUDENT_GRAPH_FULL
    ? {
      runtimeConfig: {
        shared: {
          debug: {
            logLevel: {
              defaultLogLevel: 'debug',
            },
          },
        },
        inference: {
          compute: {
            activationDtype: 'f32',
            keepF32Weights: true,
          },
        },
      },
    }
    : null;
  let student = null;
  try {
    student = await loadDistillModelHandle(studentModelRef, 'student', {
      runtime: studentRuntime,
    });
  } catch (error) {
    if (teacher?.pipeline && typeof teacher.pipeline.unload === 'function') {
      await teacher.pipeline.unload();
    }
    throw error;
  }

  const runtime = {
    stage: normalizeDistillStage(options.trainingStage || distillConfig.stage),
    teacherPipeline: teacher.pipeline,
    studentPipeline: student.pipeline,
    teacherModelId: teacher.manifest?.modelId || teacherModelRef,
    studentModelId: student.manifest?.modelId || studentModelRef,
    teacherModelUrl: teacher.modelUrl || null,
    studentModelUrl: student.modelUrl || null,
    topK: clampDistillTopK(distillConfig.topK ?? DISTILL_ADAPTER_TOP_K),
    temperature: Math.max(1e-4, toFiniteNumber(distillConfig.temperature, 1)),
    alphaKd: toFiniteNumber(distillConfig.alphaKd, 1),
    alphaCe: toFiniteNumber(distillConfig.alphaCe, 0),
    tripletMargin: Math.max(0, toFiniteNumber(distillConfig.tripletMargin, 0.2)),
    studentGraphMode,
    targetTokenMode: studentGraphMode === DISTILL_STUDENT_GRAPH_FULL
      ? 'teacher_top_token'
      : 'topk_class',
    async cleanup() {
      if (teacher?.pipeline && typeof teacher.pipeline.unload === 'function') {
        await teacher.pipeline.unload();
      }
      if (student?.pipeline && typeof student.pipeline.unload === 'function') {
        await student.pipeline.unload();
      }
    },
  };
  return runtime;
}

export function createToyModelFixture(overrides = {}) {
  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const encoderWeight = makeTensorFromFloat32(
    [0.1, -0.2, 0.3, 0.4, 0.05, -0.1],
    [3, 2],
    'training_suite_encoder_weight'
  );
  const priorWeight = makeTensorFromFloat32(
    [0.02, -0.01, 0.03, -0.05, 0.04, -0.02],
    [3, 2],
    'training_suite_prior_weight'
  );
  const decoderWeight = makeTensorFromFloat32(
    [0.03, 0.02, -0.01, 0.06, -0.04, 0.02],
    [3, 2],
    'training_suite_decoder_weight'
  );
  const baseWeight = makeTensorFromFloat32(
    [0.08, -0.12, 0.16, 0.22, -0.03, 0.09],
    [3, 2],
    'training_suite_base_weight'
  );
  const input = makeTensorFromFloat32([0.5, 0.1, -0.3, 0.2, 0.4, -0.1], [2, 3], 'training_suite_input');
  const targets = makeTensorFromUint32([1, 0], [2], 'training_suite_targets');
  const batch = { input, targets };

  const model = {
    async forward(inputTensor, tape) {
      return tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, 2, 2, 3, { transposeB: false, outputDtype: 'f32' }),
        [inputTensor, baseWeight],
        { M: 2, N: 2, K: 3, transposeB: false }
      );
    },
    loraParams() {
      return [baseWeight];
    },
    paramGroups() {
      return {
        encoder: [encoderWeight],
        prior: [priorWeight],
        decoder: [decoderWeight],
        base: [baseWeight],
        lora: [baseWeight],
      };
    },
  };

  return {
    config,
    model,
    batch,
    cleanup() {
      releaseTensor(encoderWeight);
      releaseTensor(priorWeight);
      releaseTensor(decoderWeight);
      releaseTensor(baseWeight);
      releaseTensor(input);
      releaseTensor(targets);
    },
  };
}

export const UL_STAGE_SET = Object.freeze(['stage1_joint', 'stage2_base']);

export const DISTILL_STAGE_SET = Object.freeze(['stage_a', 'stage_b']);

export const TRAINING_STAGE_SET = Object.freeze([...UL_STAGE_SET, ...DISTILL_STAGE_SET]);

export function normalizeTrainingStage(stage) {
  const normalized = String(stage || '').trim();
  if (!normalized) return null;
  if (!TRAINING_STAGE_SET.includes(normalized)) {
    throw new Error(`Unknown training stage "${normalized}". Expected one of: ${TRAINING_STAGE_SET.join(', ')}.`);
  }
  return normalized;
}

export function isDistillStage(stage) {
  return DISTILL_STAGE_SET.includes(String(stage || ''));
}

export function normalizeTrainingConfigOverride(value) {
  if (!value) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('trainingConfig must be an object when provided.');
  }
  return value;
}

export function buildDistillTrainingOverrides(options = {}) {
  const trainingConfig = normalizeTrainingConfigOverride(options.trainingConfig);
  const explicitStage = normalizeTrainingStage(options.trainingStage || trainingConfig?.distill?.stage);
  const distillEnabled = isDistillStage(explicitStage) || trainingConfig?.distill?.enabled === true;
  if (!distillEnabled) {
    return trainingConfig || null;
  }
  const stage = isDistillStage(explicitStage) ? explicitStage : 'stage_a';
  const distillOverride = {
    ...(trainingConfig?.distill || {}),
    enabled: true,
    stage,
    teacherModelId: options.teacherModelId ?? trainingConfig?.distill?.teacherModelId ?? null,
    studentModelId: options.studentModelId ?? trainingConfig?.distill?.studentModelId ?? null,
    datasetId: options.distillDatasetId ?? trainingConfig?.distill?.datasetId ?? null,
    datasetPath: options.distillDatasetPath ?? trainingConfig?.distill?.datasetPath ?? null,
    languagePair: options.distillLanguagePair ?? trainingConfig?.distill?.languagePair ?? null,
    sourceLangs: (
      options.distillSourceLangs
      ?? trainingConfig?.distill?.sourceLangs
      ?? null
    ),
    targetLangs: (
      options.distillTargetLangs
      ?? trainingConfig?.distill?.targetLangs
      ?? null
    ),
    pairAllowlist: (
      options.distillPairAllowlist
      ?? trainingConfig?.distill?.pairAllowlist
      ?? null
    ),
    strictPairContract: (
      options.strictPairContract === true
      || trainingConfig?.distill?.strictPairContract === true
    ),
    shardIndex: options.distillShardIndex ?? trainingConfig?.distill?.shardIndex ?? null,
    shardCount: options.distillShardCount ?? trainingConfig?.distill?.shardCount ?? null,
    resumeFrom: options.resumeFrom ?? trainingConfig?.distill?.resumeFrom ?? null,
    stageAArtifact: options.stageAArtifact ?? trainingConfig?.distill?.stageAArtifact ?? null,
    stageAArtifactHash: options.stageAArtifactHash ?? trainingConfig?.distill?.stageAArtifactHash ?? null,
    artifactDir: options.distillArtifactDir ?? trainingConfig?.distill?.artifactDir ?? 'reports/training/distill',
  };
  if (stage === 'stage_b') {
    distillOverride.freeze = {
      encoder: true,
      prior: true,
      decoder: true,
      base: false,
      lora: false,
      ...(trainingConfig?.distill?.freeze || {}),
    };
  }
  return {
    ...(trainingConfig || {}),
    distill: distillOverride,
  };
}
