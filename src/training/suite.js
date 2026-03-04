import { initDevice, getKernelCapabilities, getDevice } from '../gpu/device.js';
import { setPlatformsBaseUrl } from '../config/platforms/loader.js';
import { setRegistryUrl } from '../config/kernels/registry.js';
import { createTrainingConfig } from '../config/training-defaults.js';
import {
  runAttention,
  castF16ToF32,
  runGather,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
  runRoPE,
  runSiLURowSplit,
} from '../gpu/kernels/index.js';
import { createTensor } from '../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../memory/buffer-pool.js';
import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isWeightBuffer } from '../gpu/weight-buffer.js';
import { OpType } from './autograd.js';
import { AdamOptimizer } from './optimizer.js';
import { TrainingRunner } from './runner.js';
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { exportLoRAAdapter } from './export.js';
import { sha256Hex } from '../utils/sha256.js';
import { computeSampleStats } from '../debug/stats.js';
import { parseJsonl } from './datasets/jsonl.js';
import { initializeInference } from '../inference/test-harness.js';
import { createPipeline } from '../inference/pipelines/text.js';
import { parseManifest } from '../storage/rdrr-format.js';
import { openModelStore, loadManifestFromStore } from '../storage/shard-manager.js';

const LEGACY_BROWSER_TESTS = Object.freeze([
  'loss-forward',
  'softmax-backward',
  'cross-entropy-backward',
  'rmsnorm-backward',
  'layernorm-backward',
  'conv2d-backward',
  'matmul-backward',
  'embed-backward',
  'ebm-state-optimize',
  'ebm-recorded-bench',
  'parity-fixture',
  'training-leak-perf',
  'autograd-branching',
]);
const TRAINING_COMMAND_SCHEMA_VERSION = 1;
const DISTILL_ADAPTER_TOP_K = 64;
const DISTILL_LOGIT_FALLBACK = -80;
const DISTILL_STUDENT_GRAPH_PROJECTION = 'projection_head';
const DISTILL_STUDENT_GRAPH_FULL = 'transformer_full';

function buildSuiteSummary(suiteName, results, startTimeMs) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  for (const result of results) {
    if (result.skipped) {
      skipped++;
    } else if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }
  return {
    suite: suiteName,
    passed,
    failed,
    skipped,
    duration: Math.max(0, performance.now() - startTimeMs),
    results,
  };
}

function normalizeTrainingTestNames(names) {
  if (!Array.isArray(names)) return null;
  const normalized = names
    .map((name) => String(name || '').trim())
    .filter(Boolean);
  return normalized.length > 0 ? normalized : null;
}

function assertTrainingSchemaVersion(value) {
  if (value === undefined || value === null) {
    return TRAINING_COMMAND_SCHEMA_VERSION;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed !== TRAINING_COMMAND_SCHEMA_VERSION) {
    throw new Error(`trainingSchemaVersion must be ${TRAINING_COMMAND_SCHEMA_VERSION}.`);
  }
  return parsed;
}

function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tensor');
}

function makeTensorFromF16Bits(values, shape, label) {
  const data = values instanceof Uint16Array ? values : new Uint16Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor_f16');
  uploadData(buffer, data);
  return createTensor(buffer, 'f16', shape, label || 'train_tensor_f16');
}

function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tokens');
  uploadData(buffer, data);
  // Token tensors are wrapped as f32 by contract; kernels read the underlying u32 bytes.
  return createTensor(buffer, 'f32', shape, label || 'train_tokens');
}

function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

function normalizeOptionalString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function normalizeDistillDatasetPath(value) {
  return normalizeOptionalString(value);
}

function normalizeLangCode(value) {
  const normalized = normalizeOptionalString(value);
  if (!normalized) return null;
  const compact = normalized.toLowerCase().replace(/_/g, '-');
  if (compact.startsWith('en')) return 'en';
  if (compact.startsWith('es')) return 'es';
  return compact;
}

function normalizePairDirection(value) {
  const pair = normalizeOptionalString(value);
  if (!pair) return null;
  const normalized = pair.toLowerCase().replace(/_/g, '-');
  const parts = normalized.split('-').filter(Boolean);
  if (parts.length !== 2) return null;
  return `${normalizeLangCode(parts[0]) || parts[0]}->${normalizeLangCode(parts[1]) || parts[1]}`;
}

function resolveDistillDirection(record) {
  const pairDirection = normalizePairDirection(record?.pair);
  if (pairDirection) return pairDirection;
  const srcLang = normalizeLangCode(record?.src_lang);
  const tgtLang = normalizeLangCode(record?.tgt_lang || record?.lang);
  if (srcLang && tgtLang) {
    return `${srcLang}->${tgtLang}`;
  }
  return null;
}

function resolveStringCandidate(record, keys) {
  for (const key of keys) {
    const value = normalizeOptionalString(record?.[key]);
    if (value) return value;
  }
  return null;
}

function encodeDistillRow(record, index) {
  if (!record || typeof record !== 'object') return null;
  const source = resolveStringCandidate(record, ['source', 'query']);
  const targetPos = resolveStringCandidate(record, ['target_pos', 'target', 'pos']);
  const targetNeg = resolveStringCandidate(record, ['target_neg', 'neg']);
  if (!source || !targetPos) return null;
  const direction = resolveDistillDirection(record) || 'unknown';

  return {
    index,
    direction,
    source,
    targetPos,
    targetNeg: targetNeg || null,
  };
}

function summarizeDirectionCounts(samples) {
  const counts = {};
  for (const sample of samples) {
    const key = sample?.direction || 'unknown';
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

function resolveLanguageName(langCode) {
  const normalized = normalizeLangCode(langCode);
  if (normalized === 'en') return 'English';
  if (normalized === 'es') return 'Spanish';
  return normalized || 'target';
}

function buildDistillPrompt(sample) {
  const direction = String(sample?.direction || '').trim();
  const [srcCodeRaw, tgtCodeRaw] = direction.split('->');
  const srcCode = normalizeLangCode(srcCodeRaw) || srcCodeRaw || 'source';
  const tgtCode = normalizeLangCode(tgtCodeRaw) || tgtCodeRaw || 'target';
  const srcName = resolveLanguageName(srcCode);
  const tgtName = resolveLanguageName(tgtCode);
  const source = String(sample?.source || '').trim();
  return `Translate from ${srcName} to ${tgtName}:\n${source}\nTranslation:`;
}

function buildDistillCandidatePrompt(sample, candidate) {
  const base = buildDistillPrompt(sample);
  const text = String(candidate || '').trim();
  return text ? `${base} ${text}` : base;
}

function toFiniteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampDistillTopK(value) {
  const parsed = Math.floor(toFiniteNumber(value, DISTILL_ADAPTER_TOP_K));
  return Math.max(2, Math.min(256, parsed));
}

function normalizeDistillStudentGraphMode(value) {
  const normalized = normalizeOptionalString(value);
  if (!normalized) return DISTILL_STUDENT_GRAPH_FULL;
  const compact = normalized.toLowerCase().replace(/[-\s]/g, '_');
  if (compact === 'projection_head' || compact === 'projection') {
    return DISTILL_STUDENT_GRAPH_PROJECTION;
  }
  return DISTILL_STUDENT_GRAPH_FULL;
}

function toFloat32Array(values, label = 'values') {
  if (values instanceof Float32Array) return values;
  if (ArrayBuffer.isView(values)) {
    return new Float32Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  }
  if (Array.isArray(values)) {
    return new Float32Array(values);
  }
  throw new Error(`Distill ${label} must be an array-like float buffer.`);
}

function selectTopKIndices(logits, topK) {
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

function gatherLogitsByIndices(logits, indices, fallback = DISTILL_LOGIT_FALLBACK) {
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

function argmax(values) {
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

function softmax(values, temperature = 1) {
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

function disposePrefillSnapshot(result) {
  const cache = result?.cache;
  if (cache && typeof cache.clear === 'function') {
    cache.clear();
  }
}

function buildShuffledIndices(length, seed = 1337) {
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

function normalizeDistillStage(value) {
  const stage = String(value || '').trim();
  return stage === 'stage_b' ? 'stage_b' : 'stage_a';
}

async function computeTeacherPromptDistillFeatures(sample, prompt, runtime) {
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

function createDistillTensorDataset(samples, options = {}) {
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

async function loadDistillDatasetFromJsonl(datasetPath) {
  const normalizedPath = normalizeDistillDatasetPath(datasetPath);
  if (!normalizedPath) return null;
  if (!isNodeRuntime()) {
    throw new Error('distillDatasetPath currently requires Node runtime.');
  }

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
  const loadEncodedRows = (rawRows) => {
    const encodedRows = [];
    for (let i = 0; i < rawRows.length; i += 1) {
      const encoded = encodeDistillRow(rawRows[i], i);
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
      const encodedRows = loadEncodedRows(shardRows);
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
              const encodedRows = loadEncodedRows(shardRows);
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
  const encodedRows = loadEncodedRows(rows);
  if (encodedRows.length === 0) {
    throw new Error(`Distill dataset "${absolutePath}" has no usable rows.`);
  }

  return {
    absolutePath,
    rowCount: rows.length,
    sampleCount: encodedRows.length,
    directionCounts: summarizeDirectionCounts(encodedRows),
    createDataset(runOptions = {}) {
      return createDistillTensorDataset(encodedRows, runOptions);
    },
  };
}

function looksLikeUrl(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(String(value || '').trim());
}

function looksLikeFilesystemPath(value) {
  const normalized = String(value || '').trim();
  return normalized.startsWith('/') || normalized.startsWith('./') || normalized.startsWith('../');
}

async function resolveNodeModelUrlFromRef(modelRef) {
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

async function initializeInferenceFromStore(modelId) {
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

async function loadDistillModelHandle(modelRef, role, loadOptions = {}) {
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

function resolveDistillModelRefs(options = {}, trainingConfig = null) {
  const distillConfig = trainingConfig?.distill || {};
  return {
    teacherModelRef: normalizeOptionalString(options.teacherModelId ?? distillConfig.teacherModelId),
    studentModelRef: normalizeOptionalString(options.studentModelId ?? distillConfig.studentModelId),
  };
}

async function createDistillRuntimeContext(options = {}, trainingConfig = null) {
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

function resolveDistillDatasetPath(options = {}, trainingConfig = null) {
  return normalizeDistillDatasetPath(
    options.distillDatasetPath ?? trainingConfig?.distill?.datasetPath ?? null
  );
}

function resolveRuntimeUrl(pathname) {
  if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
    return pathname;
  }
  return new URL(pathname, import.meta.url).toString();
}

async function ensureTrainingGpuRuntime() {
  setPlatformsBaseUrl(resolveRuntimeUrl('../config/platforms/'));
  setRegistryUrl(resolveRuntimeUrl('../config/kernels/registry.json'));
  await initDevice();
}

function createToyModelFixture(overrides = {}) {
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
        (a, b) => runMatmul(a, b, 2, 2, 3, { transposeB: false }),
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

function resolveTensorDtype(value, fallback = 'f32') {
  const dtype = isWeightBuffer(value)
    ? value.dtype
    : (value?.dtype || getWeightDtype(value) || null);
  const normalized = String(dtype || '').toLowerCase();
  return normalized === 'f16' ? 'f16' : (normalized === 'f32' ? 'f32' : fallback);
}

async function ensureTrainableTensor(value, shape, label, ownedTrainables = null) {
  if (!value) {
    throw new Error(`Distill full-graph student missing required weight "${label}".`);
  }
  const registerOwned = (tensor) => {
    if (ownedTrainables instanceof Set && tensor?.buffer instanceof GPUBuffer) {
      ownedTrainables.add(tensor);
    }
    return tensor;
  };
  if (isWeightBuffer(value)) {
    if (value.dtype === 'f32') {
      return value;
    }
    if (value.dtype === 'f16') {
      const sourceShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
      const source = createTensor(value.buffer, 'f16', sourceShape, `${label}_source_f16`);
      const promoted = await castF16ToF32(source);
      return registerOwned(createTensor(promoted.buffer, 'f32', sourceShape, `${label}_trainable_f32`));
    }
    throw new Error(`Distill full-graph student weight "${label}" uses unsupported dtype "${value.dtype}".`);
  }
  if (value instanceof GPUBuffer) {
    const sourceShape = [...shape];
    const rawDtype = String(getBufferDtype(value) || 'f32').toLowerCase();
    const dtype = rawDtype === 'f16' ? 'f16' : 'f32';
    const tensor = createTensor(value, dtype, sourceShape, label);
    if (dtype === 'f16') {
      const promoted = await castF16ToF32(tensor);
      return registerOwned(createTensor(promoted.buffer, 'f32', sourceShape, `${label}_trainable_f32`));
    }
    return tensor;
  }
  if (isCpuWeightBuffer(value)) {
    const sourceShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
    const dtype = resolveTensorDtype(value, 'f32');
    if (dtype === 'f32') {
      const tensor = makeTensorFromFloat32(value.data, sourceShape, `${label}_cpu_f32`);
      return registerOwned(tensor);
    }
    if (dtype === 'f16') {
      let raw = null;
      if (value.data instanceof Uint16Array) {
        raw = value.data;
      } else if (ArrayBuffer.isView(value.data)) {
        raw = new Uint16Array(
          value.data.buffer,
          value.data.byteOffset,
          Math.floor(value.data.byteLength / 2)
        );
      } else if (value.data instanceof ArrayBuffer) {
        raw = new Uint16Array(value.data);
      }
      if (!raw) {
        throw new Error(`Distill full-graph student weight "${label}" has non-typed f16 CPU data.`);
      }
      const source = makeTensorFromF16Bits(raw, sourceShape, `${label}_cpu_f16`);
      const promoted = await castF16ToF32(source);
      releaseTensor(source);
      return registerOwned(createTensor(promoted.buffer, 'f32', sourceShape, `${label}_trainable_f32`));
    }
    throw new Error(`Distill full-graph student weight "${label}" has unsupported CPU dtype "${dtype}".`);
  }
  if (value.buffer instanceof GPUBuffer) {
    const resolvedShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
    const tensor = createTensor(
      value.buffer,
      resolveTensorDtype(value, 'f32'),
      resolvedShape,
      label
    );
    if (tensor.dtype === 'f16') {
      const promoted = await castF16ToF32(tensor);
      return registerOwned(createTensor(promoted.buffer, 'f32', resolvedShape, `${label}_trainable_f32`));
    }
    return tensor;
  }
  throw new Error(`Distill full-graph student weight "${label}" is not GPU-resident.`);
}

async function ensureNormTensor(value, hiddenSize, label, ownedTrainables = null) {
  return ensureTrainableTensor(value, [hiddenSize], label, ownedTrainables);
}

function hasTensorPayload(value) {
  if (!value) return false;
  if (value instanceof GPUBuffer) return true;
  if (isWeightBuffer(value) || isCpuWeightBuffer(value)) return true;
  if (value?.buffer instanceof GPUBuffer) return true;
  if (ArrayBuffer.isView(value) || Array.isArray(value)) return true;
  return false;
}

async function fuseGateUpTensors(gateTensor, upTensor, intermediateSize, hiddenSize, label, ownedTrainables = null) {
  const device = getDevice();
  if (!device) {
    throw new Error('Distill full-graph student requires active GPU device.');
  }
  if (gateTensor?.dtype !== 'f32' || upTensor?.dtype !== 'f32') {
    throw new Error(`Distill fused gate_up expects f32 tensors for "${label}".`);
  }
  const expectedRows = intermediateSize;
  const expectedCols = hiddenSize;
  const gateRows = Number.isFinite(gateTensor?.shape?.[0]) ? gateTensor.shape[0] : 0;
  const gateCols = Number.isFinite(gateTensor?.shape?.[1]) ? gateTensor.shape[1] : 0;
  const upRows = Number.isFinite(upTensor?.shape?.[0]) ? upTensor.shape[0] : 0;
  const upCols = Number.isFinite(upTensor?.shape?.[1]) ? upTensor.shape[1] : 0;
  if (gateRows !== expectedRows || gateCols !== expectedCols || upRows !== expectedRows || upCols !== expectedCols) {
    throw new Error(
      `Distill gate/up shape mismatch for "${label}": gate=[${gateRows},${gateCols}] up=[${upRows},${upCols}] ` +
      `expected=[${expectedRows},${expectedCols}]`
    );
  }
  const rowBytes = expectedCols * 4;
  const blockBytes = expectedRows * rowBytes;
  const fusedBuffer = acquireBuffer(blockBytes * 2, undefined, `${label}_fused`);
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(gateTensor.buffer, 0, fusedBuffer, 0, blockBytes);
  encoder.copyBufferToBuffer(upTensor.buffer, 0, fusedBuffer, blockBytes, blockBytes);
  device.queue.submit([encoder.finish()]);
  const fused = createTensor(fusedBuffer, 'f32', [expectedRows * 2, expectedCols], `${label}_fused`);
  if (ownedTrainables instanceof Set) {
    ownedTrainables.add(fused);
  }
  return fused;
}

function resolvePhasePrompts(batch, phase) {
  const distill = batch?.distill || {};
  const prompts = phase === 'positive'
    ? distill.tripletPositivePrompts
    : (phase === 'negative' ? distill.tripletNegativePrompts : distill.prompts);
  if (!Array.isArray(prompts) || prompts.length === 0) {
    throw new Error(`Distill student fixture requires distill prompts for phase "${phase}".`);
  }
  return prompts;
}

function createRowSliceTensor(inputTensor, rows, cols, rowIndex, label) {
  const device = getDevice();
  if (!device) {
    throw new Error('Distill full-graph student requires active GPU device.');
  }
  const dtype = inputTensor?.dtype === 'f16' ? 'f16' : 'f32';
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const rowBytes = cols * bytesPerElement;
  const clampedRow = Math.max(0, Math.min(rows - 1, rowIndex));
  const outputBuffer = acquireBuffer(rowBytes, undefined, label);
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    inputTensor.buffer,
    clampedRow * rowBytes,
    outputBuffer,
    0,
    rowBytes
  );
  device.queue.submit([encoder.finish()]);
  return createTensor(outputBuffer, dtype, [1, cols], label);
}

function createDistillStudentProjectionModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  if (!distillRuntime?.studentPipeline) {
    throw new Error('Distill student fixture requires distillRuntime.studentPipeline.');
  }
  const outputDim = clampDistillTopK(
    options.outputDim
    ?? options.inputDim
    ?? DISTILL_ADAPTER_TOP_K
  );
  const inferredEmbeddingDim = Math.floor(
    Number(distillRuntime.studentPipeline?.modelConfig?.hiddenSize)
  );
  const embeddingDim = Number.isInteger(options.embeddingDim) && options.embeddingDim > 0
    ? options.embeddingDim
    : (Number.isFinite(inferredEmbeddingDim) && inferredEmbeddingDim > 0
      ? inferredEmbeddingDim
      : outputDim);
  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const projectionWeights = new Float32Array(embeddingDim * outputDim);
  const projectionWeight = makeTensorFromFloat32(
    projectionWeights,
    [embeddingDim, outputDim],
    'distill_student_head_weight'
  );
  const temporaryInputs = new Set();

  async function projectEmbeddingInput(inputTensor, tape) {
    const rows = Number.isFinite(inputTensor?.shape?.[0]) ? inputTensor.shape[0] : 1;
    return tape.record(
      OpType.MATMUL,
      (a, b) => runMatmul(a, b, rows, outputDim, embeddingDim, { transposeB: false }),
      [inputTensor, projectionWeight],
      { M: rows, N: outputDim, K: embeddingDim, transposeB: false }
    );
  }

  async function buildStudentEmbeddingInput(batch, phase = 'anchor') {
    const distill = batch?.distill || {};
    const prompts = phase === 'positive'
      ? distill.tripletPositivePrompts
      : (phase === 'negative' ? distill.tripletNegativePrompts : distill.prompts);
    if (!Array.isArray(prompts) || prompts.length === 0) {
      throw new Error(`Distill student fixture requires distill prompts for phase "${phase}".`);
    }

    const rows = prompts.length;
    const features = new Float32Array(rows * embeddingDim);
    for (let row = 0; row < rows; row += 1) {
      const prompt = String(prompts[row] || '').trim();
      const studentResult = await distillRuntime.studentPipeline.prefillWithEmbedding(prompt, {
        useChatTemplate: false,
        embeddingMode: 'last',
      });
      try {
        const studentEmbedding = toFloat32Array(studentResult?.embedding, 'student embedding');
        const rowOffset = row * embeddingDim;
        const copyCount = Math.min(embeddingDim, studentEmbedding.length);
        features.set(studentEmbedding.subarray(0, copyCount), rowOffset);
      } finally {
        disposePrefillSnapshot(studentResult);
        distillRuntime.studentPipeline.reset();
      }
    }
    const inputTensor = makeTensorFromFloat32(
      features,
      [rows, embeddingDim],
      `distill_student_${phase}_embedding`
    );
    temporaryInputs.add(inputTensor);
    return inputTensor;
  }

  const model = {
    async forward(inputTensor, tape) {
      return projectEmbeddingInput(inputTensor, tape);
    },
    async forwardDistill(batch, tape, forwardOptions = {}) {
      const requestedPhase = String(forwardOptions?.phase || 'anchor').trim();
      const phase = requestedPhase === 'positive'
        ? 'positive'
        : (requestedPhase === 'negative' ? 'negative' : 'anchor');
      const inputTensor = await buildStudentEmbeddingInput(batch, phase);
      const logits = await projectEmbeddingInput(inputTensor, tape);
      return { logits };
    },
    cleanupDistillStep() {
      for (const tensor of temporaryInputs) {
        releaseTensor(tensor);
      }
      temporaryInputs.clear();
    },
    loraParams() {
      return [projectionWeight];
    },
    paramGroups() {
      return {
        encoder: [],
        prior: [],
        decoder: [],
        base: [projectionWeight],
        lora: [projectionWeight],
      };
    },
  };

  return {
    config,
    model,
    outputDim,
    embeddingDim,
    cleanup() {
      model.cleanupDistillStep();
      releaseTensor(projectionWeight);
    },
  };
}

async function createDistillStudentTransformerModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  const studentPipeline = distillRuntime?.studentPipeline || null;
  if (!studentPipeline?.modelConfig || !(studentPipeline.weights instanceof Map)) {
    throw new Error('Distill full-graph student fixture requires loaded student pipeline weights.');
  }
  const modelConfig = studentPipeline.modelConfig;
  const hiddenSize = Math.max(1, Math.floor(Number(modelConfig.hiddenSize) || 0));
  const intermediateSize = Math.max(1, Math.floor(Number(modelConfig.intermediateSize) || 0));
  const numLayers = Math.max(1, Math.floor(Number(modelConfig.numLayers) || 0));
  const numHeads = Math.max(1, Math.floor(Number(modelConfig.numHeads) || 0));
  const numKVHeads = Math.max(1, Math.floor(Number(modelConfig.numKVHeads || numHeads) || 0));
  const headDim = Math.max(1, Math.floor(Number(modelConfig.headDim) || 0));
  const vocabSize = Math.max(1, Math.floor(Number(modelConfig.vocabSize) || 0));
  const rmsNormEps = Number.isFinite(modelConfig.rmsNormEps) ? modelConfig.rmsNormEps : 1e-6;
  const hiddenActivation = String(modelConfig.hiddenActivation || 'silu').toLowerCase();
  const swigluLimit = Number.isFinite(modelConfig.swigluLimit) ? modelConfig.swigluLimit : 0;
  const useEmbeddingTranspose = modelConfig.embeddingTranspose === true;
  const tieWordEmbeddings = modelConfig.useTiedEmbeddings === true;

  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const ownedTrainables = new Set();
  const embeddingWeight = await ensureTrainableTensor(
    studentPipeline.weights.get('embed'),
    [vocabSize, hiddenSize],
    'embed',
    ownedTrainables
  );
  const lmHeadWeight = tieWordEmbeddings
    ? embeddingWeight
    : await ensureTrainableTensor(
      studentPipeline.weights.get('lm_head'),
      [vocabSize, hiddenSize],
      'lm_head',
      ownedTrainables
    );
  const finalNormWeight = await ensureNormTensor(
    studentPipeline.weights.get('final_norm'),
    hiddenSize,
    'final_norm',
    ownedTrainables
  );

  const ropeDim = Math.max(1, Math.floor(headDim / 2));
  const ropeRows = Math.max(1, Math.floor(Number(modelConfig.maxSeqLen) || 1));
  const ropeCos = await ensureTrainableTensor(
    createTensor(studentPipeline.ropeFreqsCos, 'f32', [ropeRows, ropeDim], 'rope_cos'),
    [ropeRows, ropeDim],
    'rope_cos',
    ownedTrainables
  );
  const ropeSin = await ensureTrainableTensor(
    createTensor(studentPipeline.ropeFreqsSin, 'f32', [ropeRows, ropeDim], 'rope_sin'),
    [ropeRows, ropeDim],
    'rope_sin',
    ownedTrainables
  );

  const layerParams = [];
  const layers = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
    const layerWeights = studentPipeline.weights.get(`layer_${layerIdx}`);
    if (!layerWeights) {
      throw new Error(`Distill full-graph student missing layer_${layerIdx} weights.`);
    }
    const gateUpWeight = layerWeights.gateUp || layerWeights.ffnGateUp || null;
    let layerGateUp = null;
    if (hasTensorPayload(gateUpWeight)) {
      layerGateUp = await ensureTrainableTensor(
        gateUpWeight,
        [intermediateSize * 2, hiddenSize],
        `layer_${layerIdx}.ffn_gate_up`,
        ownedTrainables
      );
    } else {
      const gateWeight = layerWeights.gate || layerWeights.ffnGate || null;
      const upWeight = layerWeights.up || layerWeights.ffnUp || null;
      if (!hasTensorPayload(gateWeight) || !hasTensorPayload(upWeight)) {
        throw new Error(
          `Distill full-graph student missing gate/up projections on layer ${layerIdx}.`
        );
      }
      const gateTensor = await ensureTrainableTensor(
        gateWeight,
        [intermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_gate`,
        ownedTrainables
      );
      const upTensor = await ensureTrainableTensor(
        upWeight,
        [intermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_up`,
        ownedTrainables
      );
      layerGateUp = await fuseGateUpTensors(
        gateTensor,
        upTensor,
        intermediateSize,
        hiddenSize,
        `layer_${layerIdx}.ffn_gate_up`,
        ownedTrainables
      );
    }
    const layer = {
      inputNorm: await ensureNormTensor(
        layerWeights.inputNorm,
        hiddenSize,
        `layer_${layerIdx}.input_norm`,
        ownedTrainables
      ),
      qProj: await ensureTrainableTensor(
        layerWeights.qProj,
        [numHeads * headDim, hiddenSize],
        `layer_${layerIdx}.q_proj`,
        ownedTrainables
      ),
      kProj: await ensureTrainableTensor(
        layerWeights.kProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.k_proj`,
        ownedTrainables
      ),
      vProj: await ensureTrainableTensor(
        layerWeights.vProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.v_proj`,
        ownedTrainables
      ),
      oProj: await ensureTrainableTensor(
        layerWeights.oProj,
        [hiddenSize, hiddenSize],
        `layer_${layerIdx}.o_proj`,
        ownedTrainables
      ),
      postAttentionNorm: layerWeights.postAttentionNorm
        ? await ensureNormTensor(
          layerWeights.postAttentionNorm,
          hiddenSize,
          `layer_${layerIdx}.post_attention_norm`,
          ownedTrainables
        )
        : null,
      gateUp: layerGateUp,
      down: await ensureTrainableTensor(
        layerWeights.down || layerWeights.ffnDown,
        [hiddenSize, intermediateSize],
        `layer_${layerIdx}.ffn_down`,
        ownedTrainables
      ),
    };
    layers.push(layer);
    layerParams.push(layer.inputNorm, layer.qProj, layer.kProj, layer.vProj, layer.oProj, layer.gateUp, layer.down);
    if (layer.postAttentionNorm) {
      layerParams.push(layer.postAttentionNorm);
    }
  }

  const encoderParams = [embeddingWeight, ...layerParams];
  const decoderParams = [finalNormWeight, lmHeadWeight];
  const baseParams = [...encoderParams, ...decoderParams];
  const temporaryInputs = new Set();

  async function buildPromptTokens(prompt) {
    const normalized = String(prompt || '').trim();
    if (!normalized) {
      throw new Error('Distill full-graph student prompt is empty.');
    }
    const tokenIds = studentPipeline.tokenizer.encode(normalized);
    if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
      throw new Error('Distill full-graph student tokenizer produced no tokens.');
    }
    const tokenTensor = makeTensorFromUint32(
      tokenIds,
      [tokenIds.length],
      'distill_student_prompt_tokens'
    );
    temporaryInputs.add(tokenTensor);
    return { tokenTensor, seqLen: tokenIds.length };
  }

  async function runTransformerPrompt(prompt, tape) {
    const { tokenTensor, seqLen } = await buildPromptTokens(prompt);
    let hidden = await tape.record(
      OpType.EMBED,
      (indices, embeddings) => runGather(
        indices,
        embeddings,
        seqLen,
        hiddenSize,
        vocabSize,
        {
          embeddingDtype: resolveTensorDtype(embeddingWeight, 'f32'),
          outputDtype: 'f32',
          transpose: useEmbeddingTranspose,
        }
      ),
      [tokenTensor, embeddingWeight],
      {
        numTokens: seqLen,
        hiddenSize,
        vocabSize,
        transpose: useEmbeddingTranspose,
        indexOffset: 0,
      }
    );

    for (let layerIdx = 0; layerIdx < layers.length; layerIdx += 1) {
      const layer = layers[layerIdx];
      const normed = await tape.record(
        OpType.RMSNORM,
        (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
          batchSize: seqLen,
          hiddenSize,
          rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
        }),
        [hidden, layer.inputNorm],
        { numTokens: seqLen, hiddenSize, eps: rmsNormEps }
      );

      const q2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.qProj],
        { M: seqLen, N: numHeads * headDim, K: hiddenSize, transposeB: 'auto' }
      );
      const k2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.kProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto' }
      );
      const v2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.vProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto' }
      );

      const q3d = createTensor(q2d.buffer, q2d.dtype, [seqLen, numHeads, headDim], `layer_${layerIdx}_q`);
      const k3d = createTensor(k2d.buffer, k2d.dtype, [seqLen, numKVHeads, headDim], `layer_${layerIdx}_k`);
      const v3d = createTensor(v2d.buffer, v2d.dtype, [seqLen, numKVHeads, headDim], `layer_${layerIdx}_v`);

      const qRope = await tape.record(
        OpType.ROPE,
        (q, cos, sin) => runRoPE(q, cos, sin, seqLen, { numHeads, headDim, startPos: 0 }),
        [q3d, ropeCos, ropeSin],
        { seqLen, numHeads, headDim, startPos: 0 }
      );
      const kRope = await tape.record(
        OpType.ROPE,
        (k, cos, sin) => runRoPE(k, cos, sin, seqLen, { numHeads: numKVHeads, headDim, startPos: 0 }),
        [k3d, ropeCos, ropeSin],
        { seqLen, numHeads: numKVHeads, headDim, startPos: 0 }
      );

      const attention = await tape.record(
        OpType.ATTENTION,
        (q, k, v) => runAttention(q, k, v, null, numHeads, headDim, {
          seqLen,
          kvLen: seqLen,
          numKVHeads,
          causal: true,
          startPos: 0,
          scale: 1 / Math.sqrt(headDim),
        }),
        [qRope, kRope, v3d],
        { seqLen, numHeads, headDim, scale: 1 / Math.sqrt(headDim), causal: true, recomputeForward: true }
      );
      const attention2d = createTensor(
        attention.buffer,
        attention.dtype,
        [seqLen, hiddenSize],
        `layer_${layerIdx}_attn_2d`
      );

      const attentionOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [attention2d, layer.oProj],
        { M: seqLen, N: hiddenSize, K: hiddenSize, transposeB: 'auto' }
      );
      const postAttention = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [attentionOutput, hidden],
        { size: seqLen * hiddenSize }
      );

      const ffnInput = layer.postAttentionNorm
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [postAttention, layer.postAttentionNorm],
          { numTokens: seqLen, hiddenSize, eps: rmsNormEps }
        )
        : postAttention;
      const gateUp = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, intermediateSize * 2, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [ffnInput, layer.gateUp],
        { M: seqLen, N: intermediateSize * 2, K: hiddenSize, transposeB: 'auto' }
      );
      const activated = await tape.record(
        OpType.SILU_ROWSPLIT,
        (x) => runSiLURowSplit(x, {
          numTokens: seqLen,
          dim: intermediateSize,
          activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
          swigluLimit: hiddenActivation === 'gelu' ? null : swigluLimit,
        }),
        [gateUp],
        {
          numTokens: seqLen,
          dim: intermediateSize,
          activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
          swigluLimit: hiddenActivation === 'gelu' ? 0 : swigluLimit,
        }
      );
      const ffnOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, intermediateSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [activated, layer.down],
        { M: seqLen, N: hiddenSize, K: intermediateSize, transposeB: 'auto' }
      );
      hidden = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [ffnOutput, postAttention],
        { size: seqLen * hiddenSize }
      );
    }

    const finalHidden = await tape.record(
      OpType.RMSNORM,
      (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
        batchSize: seqLen,
        hiddenSize,
        rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
      }),
      [hidden, finalNormWeight],
      { numTokens: seqLen, hiddenSize, eps: rmsNormEps }
    );
    const lastHidden = await tape.record(
      OpType.ROW_SLICE,
      (x) => createRowSliceTensor(x, seqLen, hiddenSize, seqLen - 1, 'distill_last_hidden'),
      [finalHidden],
      { rows: seqLen, cols: hiddenSize, rowIndex: seqLen - 1 }
    );
    return tape.record(
      OpType.MATMUL,
      (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
        transposeB: 'auto',
        outputDtype: 'f32',
      }),
      [lastHidden, lmHeadWeight],
      { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto' }
    );
  }

  const model = {
    async forward(inputTensor, tape) {
      return tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [inputTensor, lmHeadWeight],
        { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto' }
      );
    },
    async forwardDistill(batch, tape, forwardOptions = {}) {
      const requestedPhase = String(forwardOptions?.phase || 'anchor').trim();
      const phase = requestedPhase === 'positive'
        ? 'positive'
        : (requestedPhase === 'negative' ? 'negative' : 'anchor');
      const prompts = resolvePhasePrompts(batch, phase);
      if (prompts.length !== 1) {
        throw new Error(
          `Distill full-graph student currently requires batchSize=1, got ${prompts.length}.`
        );
      }
      const logits = await runTransformerPrompt(prompts[0], tape);
      return { logits };
    },
    cleanupDistillStep() {
      for (const tensor of temporaryInputs) {
        releaseTensor(tensor);
      }
      temporaryInputs.clear();
    },
    loraParams() {
      return decoderParams;
    },
    paramGroups() {
      return {
        encoder: encoderParams,
        prior: [],
        decoder: decoderParams,
        base: baseParams,
        lora: [],
      };
    },
  };

  return {
    config,
    model,
    outputDim: vocabSize,
    embeddingDim: hiddenSize,
    cleanup() {
      model.cleanupDistillStep();
      for (const tensor of ownedTrainables) {
        releaseTensor(tensor);
      }
      ownedTrainables.clear();
    },
  };
}

async function createDistillStudentRuntimeModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  const graphMode = normalizeDistillStudentGraphMode(
    options.studentGraphMode
    ?? distillRuntime?.studentGraphMode
    ?? overrides?.training?.distill?.studentGraphMode
  );
  if (graphMode === DISTILL_STUDENT_GRAPH_PROJECTION) {
    return createDistillStudentProjectionModelFixture(overrides, options);
  }
  return createDistillStudentTransformerModelFixture(overrides, options);
}

async function runRunnerSmokeTest() {
  const fixture = createToyModelFixture();
  try {
    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const dataset = {
      async *batches() {
        for (let i = 0; i < 3; i += 1) {
          yield fixture.batch;
        }
      },
    };

    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 3,
    });
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return { passed: false, error: 'Training runner produced no metrics.' };
    }
    for (const entry of metrics) {
      if (!isFiniteNumber(entry.total_loss) || !isFiniteNumber(entry.step_time_ms)) {
        return { passed: false, error: 'Training runner emitted non-finite metrics.' };
      }
    }

    return { passed: true };
  } finally {
    fixture.cleanup();
  }
}

async function runTrainStepMetricsTest() {
  const fixture = createToyModelFixture();
  try {
    const result = await trainStep(fixture.model, fixture.batch, fixture.config, {
      crossEntropyLoss,
      clipGradients,
      optimizer: new AdamOptimizer(fixture.config),
    });

    if (!isFiniteNumber(result.forward_ms) || !isFiniteNumber(result.backward_ms)) {
      return { passed: false, error: 'trainStep did not report finite phase timings.' };
    }
    if (!result.clipMetrics || !isFiniteNumber(result.clipMetrics.gradient_norm_unclipped)) {
      return { passed: false, error: 'trainStep did not report clipping metrics.' };
    }
    if (!result.optimizerMetrics || !isFiniteNumber(result.optimizerMetrics.optimizer_ms)) {
      return { passed: false, error: 'trainStep did not report optimizer metrics.' };
    }

    return { passed: true };
  } finally {
    fixture.cleanup();
  }
}

const UL_STAGE_SET = Object.freeze(['stage1_joint', 'stage2_base']);
const DISTILL_STAGE_SET = Object.freeze(['stage_a', 'stage_b']);
const TRAINING_STAGE_SET = Object.freeze([...UL_STAGE_SET, ...DISTILL_STAGE_SET]);

function normalizeTrainingStage(stage) {
  const normalized = String(stage || '').trim();
  if (!normalized) return null;
  if (!TRAINING_STAGE_SET.includes(normalized)) {
    throw new Error(`Unknown training stage "${normalized}". Expected one of: ${TRAINING_STAGE_SET.join(', ')}.`);
  }
  return normalized;
}

function isUlStage(stage) {
  return UL_STAGE_SET.includes(String(stage || ''));
}

function isDistillStage(stage) {
  return DISTILL_STAGE_SET.includes(String(stage || ''));
}

function normalizeTrainingConfigOverride(value) {
  if (!value) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('trainingConfig must be an object when provided.');
  }
  return value;
}

function normalizeAdapterActivationConfig(options = {}) {
  const runtimeConfig = normalizeTrainingConfigOverride(options.trainingConfig);
  const direct = options.adapterActivation;
  const nested = runtimeConfig?.adapterActivation;
  const config = direct && typeof direct === 'object' ? direct : (nested && typeof nested === 'object' ? nested : null);
  if (!config) {
    return {
      enabled: false,
      autoActivate: false,
      adapterPayload: null,
      exportConfig: null,
    };
  }
  const exportConfig = config.export && typeof config.export === 'object'
    ? config.export
    : null;
  const adapterPayload = (() => {
    if (config.adapterManifest && typeof config.adapterManifest === 'object') {
      return { adapterManifest: config.adapterManifest };
    }
    if (typeof config.adapterManifestJson === 'string' && config.adapterManifestJson.trim()) {
      return { adapterManifestJson: config.adapterManifestJson };
    }
    if (typeof config.adapterManifestUrl === 'string' && config.adapterManifestUrl.trim()) {
      return { adapterManifestUrl: config.adapterManifestUrl };
    }
    if (typeof config.adapterManifestPath === 'string' && config.adapterManifestPath.trim()) {
      return { adapterManifestPath: config.adapterManifestPath };
    }
    if (config.adapter != null) {
      return { adapter: config.adapter };
    }
    return null;
  })();
  return {
    enabled: config.enabled !== false,
    autoActivate: config.autoActivate === true,
    adapterPayload,
    exportConfig,
  };
}

function normalizeLoRAExportConfig(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const tensors = Array.isArray(value.tensors) ? value.tensors : [];
  if (tensors.length === 0) {
    return null;
  }
  const normalizedTensors = tensors.map((entry, index) => {
    const name = normalizeOptionalString(entry?.name);
    const paramIndex = Number.isFinite(entry?.paramIndex)
      ? Math.floor(entry.paramIndex)
      : -1;
    if (!name) {
      throw new Error(`adapterActivation.export.tensors[${index}].name is required.`);
    }
    if (!Number.isInteger(paramIndex) || paramIndex < 0) {
      throw new Error(`adapterActivation.export.tensors[${index}].paramIndex must be a non-negative integer.`);
    }
    return { name, paramIndex };
  });
  const targetModules = Array.isArray(value.targetModules)
    ? value.targetModules.map((moduleName) => String(moduleName || '').trim()).filter(Boolean)
    : [];
  if (targetModules.length === 0) {
    throw new Error('adapterActivation.export.targetModules must contain at least one module.');
  }
  const id = normalizeOptionalString(value.id);
  const name = normalizeOptionalString(value.name);
  const baseModel = normalizeOptionalString(value.baseModel);
  const rank = Number(value.rank);
  const alpha = Number(value.alpha);
  if (!id || !name || !baseModel) {
    throw new Error('adapterActivation.export requires id, name, and baseModel.');
  }
  if (!Number.isFinite(rank) || rank <= 0 || !Number.isInteger(rank)) {
    throw new Error('adapterActivation.export.rank must be a positive integer.');
  }
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new Error('adapterActivation.export.alpha must be a positive number.');
  }
  return {
    id,
    name,
    baseModel,
    rank,
    alpha,
    targetModules,
    tensors: normalizedTensors,
    format: value.format === 'array' ? 'array' : 'base64',
    pretty: value.pretty === true,
  };
}

async function exportLoRAAdapterFromModel(model, exportConfig, runIndex = null) {
  const normalizedConfig = normalizeLoRAExportConfig(exportConfig);
  if (!normalizedConfig) return null;
  if (!model || typeof model.loraParams !== 'function') {
    throw new Error('adapterActivation.export requires model.loraParams() support.');
  }
  const params = model.loraParams();
  if (!Array.isArray(params) || params.length === 0) {
    throw new Error('adapterActivation.export requires non-empty model.loraParams().');
  }
  const tensors = normalizedConfig.tensors.map((entry) => {
    const tensor = params[entry.paramIndex];
    if (!tensor) {
      throw new Error(`adapterActivation.export tensor paramIndex ${entry.paramIndex} is out of range.`);
    }
    return {
      name: entry.name,
      tensor,
    };
  });
  const exported = await exportLoRAAdapter({
    id: normalizedConfig.id,
    name: normalizedConfig.name,
    baseModel: normalizedConfig.baseModel,
    rank: normalizedConfig.rank,
    alpha: normalizedConfig.alpha,
    targetModules: normalizedConfig.targetModules,
    tensors,
    format: normalizedConfig.format,
    pretty: normalizedConfig.pretty,
  });
  return {
    runIndex,
    manifest: exported.manifest,
    json: exported.json,
    hash: sha256Hex(exported.json),
  };
}

async function tryActivateAdapterPayload(payload) {
  if (!payload) {
    return {
      activated: false,
      adapterName: null,
      source: null,
      reason: 'no_adapter_payload',
    };
  }
  const { activateLoRAFromTrainingOutput } = await import('../client/doppler-provider/model-manager.js');
  try {
    return await activateLoRAFromTrainingOutput(payload);
  } catch (error) {
    return {
      activated: false,
      adapterName: null,
      source: null,
      reason: String(error?.message || error),
    };
  }
}

function buildUlTrainingOverrides(options = {}) {
  const trainingConfig = normalizeTrainingConfigOverride(options.trainingConfig);
  const explicitStage = normalizeTrainingStage(options.trainingStage || trainingConfig?.ul?.stage);
  const ulEnabled = isUlStage(explicitStage) || trainingConfig?.ul?.enabled === true;
  if (!ulEnabled) {
    return trainingConfig || null;
  }
  const stage = isUlStage(explicitStage) ? explicitStage : 'stage1_joint';
  const ulOverride = {
    ...(trainingConfig?.ul || {}),
    enabled: true,
    stage,
    stage1Artifact: options.stage1Artifact ?? trainingConfig?.ul?.stage1Artifact ?? null,
    stage1ArtifactHash: options.stage1ArtifactHash ?? trainingConfig?.ul?.stage1ArtifactHash ?? null,
    artifactDir: options.ulArtifactDir ?? trainingConfig?.ul?.artifactDir ?? 'reports/training/ul',
  };
  if (stage === 'stage2_base') {
    ulOverride.freeze = {
      encoder: true,
      prior: true,
      decoder: true,
      base: false,
      lora: false,
      ...(trainingConfig?.ul?.freeze || {}),
    };
  }
  return {
    ...(trainingConfig || {}),
    ul: ulOverride,
  };
}

function buildDistillTrainingOverrides(options = {}) {
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

async function computeNodeFileHash(filePath) {
  if (!(typeof process !== 'undefined' && process.versions?.node)) {
    return null;
  }
  const [{ readFile }, { resolve }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
  ]);
  const absolutePath = resolve(String(filePath));
  const raw = await readFile(absolutePath, 'utf8');
  return {
    absolutePath,
    hash: sha256Hex(raw),
  };
}

async function resolveIsolatedArtifactDir(explicitDir, prefix) {
  const normalized = normalizeOptionalString(explicitDir);
  if (normalized) {
    return normalized;
  }
  if (!(typeof process !== 'undefined' && process.versions?.node)) {
    return null;
  }
  const [{ mkdtemp }, { tmpdir }, { join }] = await Promise.all([
    import('node:fs/promises'),
    import('node:os'),
    import('node:path'),
  ]);
  return mkdtemp(join(tmpdir(), `doppler-${prefix}-`));
}

async function runUlStageTest(stage, options = {}) {
  const ulTraining = buildUlTrainingOverrides({
    ...options,
    trainingStage: stage,
  });
  const fixture = createToyModelFixture({
    training: ulTraining || undefined,
  });

  try {
    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const dataset = {
      async *batches() {
        for (let i = 0; i < 2; i += 1) {
          yield fixture.batch;
        }
      },
    };
    const ulArtifactDir = await resolveIsolatedArtifactDir(options.ulArtifactDir, 'ul');
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 2,
      modelId: options.modelId || 'training',
      modelUrl: options.modelUrl || null,
      timestamp: options.timestamp || null,
      ulArtifactDir,
    });
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return { passed: false, error: `UL ${stage} produced no metrics.` };
    }
    const requiredFields = [
      'loss_prior',
      'loss_decoder',
      'loss_recon',
      'lambda',
      'latent_bitrate_proxy',
      'loss_total',
      'coeff_ce',
      'coeff_prior',
      'coeff_decoder',
      'coeff_recon',
    ];
    if (stage === 'stage1_joint') {
      requiredFields.push(
        'schedule_step_index',
        'latent_clean_mean',
        'latent_clean_std',
        'latent_noise_mean',
        'latent_noise_std',
        'latent_noisy_mean',
        'latent_noisy_std',
        'latent_shape',
        'latent_clean_values',
        'latent_noise_values',
        'latent_noisy_values'
      );
    }
    if (stage === 'stage2_base') {
      requiredFields.push('stage1_latent_count');
    }
    for (const field of requiredFields) {
      if (!(field in metrics[0])) {
        return { passed: false, error: `UL ${stage} missing metric field "${field}".` };
      }
    }
    const artifact = runner.lastArtifact;
    if (!artifact || !artifact.manifestPath) {
      return { passed: false, error: `UL ${stage} did not produce artifacts.` };
    }
    return {
      passed: true,
      artifact,
      metrics: {
        stage,
        steps: metrics.length,
        manifestPath: artifact.manifestPath,
        manifestHash: artifact.manifestHash,
        manifestContentHash: artifact.manifestContentHash,
        manifestFileHash: artifact.manifestFileHash ?? null,
        ulResolvedConfig: {
          enabled: fixture.config.training?.ul?.enabled === true,
          stage: fixture.config.training?.ul?.stage ?? null,
          lambda0: fixture.config.training?.ul?.lambda0 ?? null,
          seed: fixture.config.training?.ul?.seed ?? null,
          noiseSchedule: fixture.config.training?.ul?.noiseSchedule ?? null,
          priorAlignment: fixture.config.training?.ul?.priorAlignment ?? null,
          decoderSigmoidWeight: fixture.config.training?.ul?.decoderSigmoidWeight ?? null,
          freeze: fixture.config.training?.ul?.freeze ?? null,
        },
      },
    };
  } finally {
    fixture.cleanup();
  }
}

async function runUlStage1Test(options = {}) {
  return runUlStageTest('stage1_joint', options);
}

async function runUlStage2Test(options = {}) {
  const explicitStage1Artifact = String(options.stage1Artifact || '').trim();
  let stage1Artifact = explicitStage1Artifact || null;
  let stage1ArtifactHash = String(options.stage1ArtifactHash || '').trim() || null;

  if (!stage1Artifact) {
    const stage1 = await runUlStage1Test({
      ...options,
      trainingStage: 'stage1_joint',
    });
    if (!stage1?.passed || !stage1?.artifact?.manifestPath) {
      return { passed: false, error: 'UL stage2 preflight failed to generate stage1 artifact.' };
    }
    stage1Artifact = stage1.artifact.manifestPath;
    stage1ArtifactHash = stage1.artifact.manifestHash;
    const nodeHash = await computeNodeFileHash(stage1Artifact);
    if (nodeHash?.hash) {
      stage1ArtifactHash = nodeHash.hash;
      stage1Artifact = nodeHash.absolutePath;
    }
  }

  return runUlStageTest('stage2_base', {
    ...options,
    stage1Artifact,
    stage1ArtifactHash,
  });
}

async function runDistillStageTest(stage, options = {}) {
  const distillTraining = buildDistillTrainingOverrides({
    ...options,
    trainingStage: stage,
  });
  const distillOutputDim = clampDistillTopK(distillTraining?.distill?.topK ?? DISTILL_ADAPTER_TOP_K);
  const resolvedTrainingConfig = createTrainingConfig({
    training: distillTraining || undefined,
  }).training;
  let fixture = null;
  let distillRuntime = null;

  try {
    const distillDatasetPath = resolveDistillDatasetPath(options, resolvedTrainingConfig);
    if (!distillDatasetPath) {
      throw new Error('Distill stage requires --distill-dataset-path (training.distill.datasetPath).');
    }
    const distillDatasetReport = await loadDistillDatasetFromJsonl(distillDatasetPath);
    distillRuntime = await createDistillRuntimeContext({
      ...options,
      trainingStage: stage,
    }, resolvedTrainingConfig);
    fixture = await createDistillStudentRuntimeModelFixture({
      training: distillTraining || undefined,
    }, {
      outputDim: distillOutputDim,
      distillRuntime,
    });

    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const distillMaxSteps = Number.isInteger(options.trainingBenchSteps) && options.trainingBenchSteps > 0
      ? options.trainingBenchSteps
      : 2;
    const dataset = distillDatasetReport.createDataset({
      batchSize: 1,
      shuffle: false,
      seed: 1337,
      distillRuntime,
    });
    const distillRunStartMs = performance.now();
    const distillArtifactDir = await resolveIsolatedArtifactDir(options.distillArtifactDir, 'distill');
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: distillMaxSteps,
      modelId: options.modelId || distillRuntime.studentModelId || 'training',
      modelUrl: options.modelUrl || distillRuntime.studentModelUrl || null,
      timestamp: options.timestamp || null,
      distillArtifactDir,
      stageAArtifact: options.stageAArtifact || null,
      stageAArtifactHash: options.stageAArtifactHash || null,
      teacherModelId: distillRuntime.teacherModelId || null,
      studentModelId: distillRuntime.studentModelId || null,
      distillDatasetId: options.distillDatasetId || null,
      distillDatasetPath: distillDatasetReport.absolutePath,
      distillLanguagePair: options.distillLanguagePair || null,
      distillShardIndex: options.distillShardIndex ?? fixture.config.training?.distill?.shardIndex ?? null,
      distillShardCount: options.distillShardCount ?? fixture.config.training?.distill?.shardCount ?? null,
      resumeFrom: options.resumeFrom ?? fixture.config.training?.distill?.resumeFrom ?? null,
    });
    if (!Array.isArray(metrics) || metrics.length === 0) {
      return { passed: false, error: `Distill ${stage} produced no metrics.` };
    }
    const requiredFields = stage === 'stage_a'
      ? ['loss_kd', 'distill_stage']
      : ['loss_triplet', 'distill_stage', 'distill_triplet_margin'];
    for (const field of requiredFields) {
      if (!(field in metrics[0])) {
        return { passed: false, error: `Distill ${stage} missing metric field "${field}".` };
      }
    }
    const artifact = runner.lastArtifact;
    if (!artifact || !artifact.manifestPath) {
      return { passed: false, error: `Distill ${stage} did not produce artifacts.` };
    }
    const progress = resolveBenchProgressSummary(
      metrics,
      resolveDistillShardProgressContext(
        options,
        fixture.config.training,
        distillMaxSteps,
        distillDatasetReport?.shardCount ?? null
      ),
      distillRunStartMs
    );
    return {
      passed: true,
      artifact,
      metrics: {
        stage,
        steps: metrics.length,
        progress,
        manifestPath: artifact.manifestPath,
        manifestHash: artifact.manifestHash,
        manifestContentHash: artifact.manifestContentHash,
        manifestFileHash: artifact.manifestFileHash ?? null,
        distillResolvedConfig: {
          enabled: fixture.config.training?.distill?.enabled === true,
          stage: fixture.config.training?.distill?.stage ?? null,
          teacherModelId: fixture.config.training?.distill?.teacherModelId ?? null,
          studentModelId: fixture.config.training?.distill?.studentModelId ?? null,
          datasetId: fixture.config.training?.distill?.datasetId ?? null,
          datasetPath: fixture.config.training?.distill?.datasetPath ?? null,
          languagePair: fixture.config.training?.distill?.languagePair ?? null,
          shardIndex: fixture.config.training?.distill?.shardIndex ?? null,
          shardCount: fixture.config.training?.distill?.shardCount ?? null,
          resumeFrom: fixture.config.training?.distill?.resumeFrom ?? null,
          temperature: fixture.config.training?.distill?.temperature ?? null,
          alphaKd: fixture.config.training?.distill?.alphaKd ?? null,
          alphaCe: fixture.config.training?.distill?.alphaCe ?? null,
          tripletMargin: fixture.config.training?.distill?.tripletMargin ?? null,
          studentGraphMode: fixture.config.training?.distill?.studentGraphMode ?? null,
          topK: fixture.config.training?.distill?.topK ?? distillOutputDim,
          freeze: fixture.config.training?.distill?.freeze ?? null,
        },
        distillRuntime: {
          teacherModelId: distillRuntime.teacherModelId || null,
          studentModelId: distillRuntime.studentModelId || null,
          teacherModelUrl: distillRuntime.teacherModelUrl || null,
          studentModelUrl: distillRuntime.studentModelUrl || null,
          topK: distillRuntime.topK,
          studentGraphMode: distillRuntime.studentGraphMode || null,
          targetTokenMode: distillRuntime.targetTokenMode || null,
        },
        distillDataset: {
          path: distillDatasetReport.absolutePath,
          rowCount: distillDatasetReport.rowCount,
          sampleCount: distillDatasetReport.sampleCount,
          shardCount: distillDatasetReport.shardCount ?? 1,
          directionCounts: distillDatasetReport.directionCounts,
        },
        checkpoint: runner.lastCheckpoint || null,
      },
    };
  } finally {
    if (distillRuntime && typeof distillRuntime.cleanup === 'function') {
      await distillRuntime.cleanup();
    }
    if (fixture) {
      fixture.cleanup();
    }
  }
}

async function runDistillStageATest(options = {}) {
  return runDistillStageTest('stage_a', options);
}

async function runDistillStageBTest(options = {}) {
  const explicitStageAArtifact = String(options.stageAArtifact || '').trim();
  let stageAArtifact = explicitStageAArtifact || null;
  let stageAArtifactHash = String(options.stageAArtifactHash || '').trim() || null;

  if (!stageAArtifact) {
    const stageA = await runDistillStageATest({
      ...options,
      trainingStage: 'stage_a',
    });
    if (!stageA?.passed || !stageA?.artifact?.manifestPath) {
      return { passed: false, error: 'Distill stage_b preflight failed to generate stage_a artifact.' };
    }
    stageAArtifact = stageA.artifact.manifestPath;
    stageAArtifactHash = stageA.artifact.manifestHash;
    const nodeHash = await computeNodeFileHash(stageAArtifact);
    if (nodeHash?.hash) {
      stageAArtifactHash = nodeHash.hash;
      stageAArtifact = nodeHash.absolutePath;
    }
  }

  return runDistillStageTest('stage_b', {
    ...options,
    stageAArtifact,
    stageAArtifactHash,
  });
}

function createLegacySkippedTest(name) {
  return async () => ({
    passed: true,
    skipped: true,
    error: `Legacy browser-only test "${name}" remains in tests/training/browser/test-page.js.`,
  });
}

const CORE_TESTS = Object.freeze({
  'runner-smoke': runRunnerSmokeTest,
  'train-step-metrics': runTrainStepMetricsTest,
  'ul-stage1': runUlStage1Test,
  'ul-stage2': runUlStage2Test,
  'distill-stage-a': runDistillStageATest,
  'distill-stage-b': runDistillStageBTest,
});

const TESTS = Object.freeze({
  ...CORE_TESTS,
  ...Object.fromEntries(LEGACY_BROWSER_TESTS.map((name) => [name, createLegacySkippedTest(name)])),
});

export const trainingHarness = Object.freeze({
  async getGPU() {
    await ensureTrainingGpuRuntime();
    return true;
  },
  async runTest(name, options = {}) {
    const fn = TESTS[name];
    if (!fn) {
      return { passed: false, error: `Unknown training test: ${name}` };
    }
    return fn(options);
  },
  listTests() {
    return Object.keys(TESTS);
  },
});

export async function runTrainingSuite(options = {}) {
  const trainingSchemaVersion = assertTrainingSchemaVersion(options.trainingSchemaVersion);
  const adapterActivation = normalizeAdapterActivationConfig(options);
  const startTime = performance.now();
  await trainingHarness.getGPU();

  const availableTests = trainingHarness.listTests();
  const requestedTestsFromOptions = normalizeTrainingTestNames(options.trainingTests);
  const requestedStage = normalizeTrainingStage(options.trainingStage);
  const stageDefaultTests = requestedStage === 'stage1_joint'
    ? ['ul-stage1']
    : (
      requestedStage === 'stage2_base'
        ? ['ul-stage2']
        : (
          requestedStage === 'stage_a'
            ? ['distill-stage-a']
            : (requestedStage === 'stage_b' ? ['distill-stage-b'] : null)
        )
    );
  const requestedTests = requestedTestsFromOptions || stageDefaultTests;
  if (requestedTests) {
    const unknownTests = requestedTests.filter((name) => !availableTests.includes(name));
    if (unknownTests.length > 0) {
      throw new Error(`Unknown training test(s): ${unknownTests.join(', ')}`);
    }
  }
  const testsToRun = requestedTests ?? availableTests;

  const results = [];
  for (const testName of testsToRun) {
    const testStart = performance.now();
    try {
      const outcome = await trainingHarness.runTest(testName, options);
      const passed = outcome?.passed === true;
      const skipped = outcome?.skipped === true;
      const errorMessage = skipped
        ? (outcome?.error ? String(outcome.error) : undefined)
        : (passed ? undefined : String(outcome?.error || 'Training test failed'));
      const entry = {
        name: testName,
        passed,
        skipped,
        duration: Math.max(0, performance.now() - testStart),
        ...(errorMessage ? { error: errorMessage } : {}),
      };
      if (outcome?.metrics && typeof outcome.metrics === 'object') {
        entry.metrics = outcome.metrics;
      }
      if (outcome?.artifact && typeof outcome.artifact === 'object') {
        entry.artifact = outcome.artifact;
      }
      results.push(entry);
    } catch (error) {
      results.push({
        name: testName,
        passed: false,
        duration: Math.max(0, performance.now() - testStart),
        error: String(error?.message || error),
      });
    }
  }

  const summary = buildSuiteSummary('training', results, startTime);
  const adapterActivationResult = (
    adapterActivation.enabled
    && adapterActivation.autoActivate
  )
    ? await tryActivateAdapterPayload(adapterActivation.adapterPayload)
    : null;
  return {
    ...summary,
    modelId: options.modelId || options.modelUrl || 'training',
    metrics: {
      testsRun: results.length,
      selectedTests: testsToRun,
      availableTests,
      trainingStage: requestedStage || null,
      trainingSchemaVersion,
      adapterActivation: adapterActivationResult,
    },
    deviceInfo: getKernelCapabilities(),
  };
}

function toPositiveInteger(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const floored = Math.floor(parsed);
  return floored > 0 ? floored : fallback;
}

function toPositiveIntegerOrNull(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  const floored = Math.floor(parsed);
  return floored > 0 ? floored : null;
}

function resolveDistillShardProgressContext(
  options = {},
  trainingOverrides = null,
  stepsPerShard = null,
  fallbackShardCount = null
) {
  const distillConfig = trainingOverrides?.distill || {};
  const shardIndexInput = toPositiveIntegerOrNull(
    options.distillShardIndex ?? distillConfig.shardIndex ?? null
  );
  const shardCountInput = toPositiveIntegerOrNull(
    options.distillShardCount ?? distillConfig.shardCount ?? null
  );
  const fallbackShardCountInput = toPositiveIntegerOrNull(fallbackShardCount);
  if (
    shardIndexInput !== null
    && shardCountInput !== null
    && shardIndexInput > shardCountInput
  ) {
    throw new Error('distillShardIndex must be <= distillShardCount.');
  }
  const shardCount = shardCountInput ?? fallbackShardCountInput ?? 1;
  const shardIndex = shardIndexInput ?? 1;
  const normalizedStepsPerShard = toPositiveIntegerOrNull(stepsPerShard);
  const totalGlobalSteps = normalizedStepsPerShard
    ? (normalizedStepsPerShard * shardCount)
    : null;
  return {
    shardIndex: Math.min(Math.max(1, shardIndex), shardCount),
    shardCount: Math.max(1, shardCount),
    stepsPerShard: normalizedStepsPerShard,
    totalGlobalSteps,
  };
}

function resolveBenchProgressSummary(stepEntries, context, startTimeMs) {
  const entries = Array.isArray(stepEntries) ? stepEntries : [];
  const lastEntry = entries.length > 0 ? entries[entries.length - 1] : null;
  const shardIndex = context?.shardIndex ?? 1;
  const shardCount = context?.shardCount ?? 1;
  const stepsPerShard = context?.stepsPerShard ?? null;
  const totalGlobalSteps = context?.totalGlobalSteps ?? null;
  const fallbackGlobalStep = stepsPerShard
    ? (((shardIndex - 1) * stepsPerShard) + Math.min(entries.length, stepsPerShard))
    : null;
  const completedGlobalSteps = Number.isFinite(lastEntry?.progress_global_step)
    ? lastEntry.progress_global_step
    : fallbackGlobalStep;
  const resolvedTotalGlobalSteps = Number.isFinite(lastEntry?.progress_global_steps)
    ? lastEntry.progress_global_steps
    : totalGlobalSteps;
  const percentComplete = Number.isFinite(lastEntry?.progress_percent_complete)
    ? lastEntry.progress_percent_complete
    : (
      Number.isFinite(completedGlobalSteps)
      && Number.isFinite(resolvedTotalGlobalSteps)
      && resolvedTotalGlobalSteps > 0
      ? Math.min(100, (completedGlobalSteps / resolvedTotalGlobalSteps) * 100)
      : null
    );
  const etaMs = Number.isFinite(lastEntry?.progress_eta_ms)
    ? Math.max(0, lastEntry.progress_eta_ms)
    : (
      Number.isFinite(percentComplete)
      && percentComplete >= 100
      ? 0
      : null
    );
  const elapsedMs = Number.isFinite(lastEntry?.progress_elapsed_ms)
    ? Math.max(0, lastEntry.progress_elapsed_ms)
    : Math.max(0, performance.now() - startTimeMs);
  return {
    shardIndex,
    shardCount,
    stepsPerShard,
    completedGlobalSteps: Number.isFinite(completedGlobalSteps) ? completedGlobalSteps : null,
    totalGlobalSteps: Number.isFinite(resolvedTotalGlobalSteps) ? resolvedTotalGlobalSteps : null,
    percentComplete,
    etaMs,
    etaIso: Number.isFinite(etaMs) ? new Date(Date.now() + etaMs).toISOString() : null,
    elapsedMs,
    updatedAt: new Date().toISOString(),
  };
}

function appendTimelineEvent(timeline, type, details = {}) {
  timeline.push({
    index: timeline.length + 1,
    timestamp: new Date().toISOString(),
    type,
    ...details,
  });
}

function resolveBenchRunSettings(options = {}) {
  const benchRun = options.benchRun && typeof options.benchRun === 'object'
    ? options.benchRun
    : {};
  return {
    warmupRuns: Math.max(0, Math.floor(Number(benchRun.warmupRuns) || 0)),
    timedRuns: toPositiveInteger(benchRun.timedRuns, 1),
    stepsPerRun: toPositiveInteger(
      options.trainingBenchSteps ?? benchRun.steps ?? options.trainingSteps,
      2
    ),
  };
}

function resolveTrainingOverrides(options = {}) {
  const distillTraining = buildDistillTrainingOverrides(options);
  if (distillTraining?.distill?.enabled) {
    return distillTraining;
  }
  const ulTraining = buildUlTrainingOverrides(options);
  if (ulTraining) {
    return ulTraining;
  }
  return normalizeTrainingConfigOverride(options.trainingConfig) || undefined;
}

export async function runTrainingBenchSuite(options = {}) {
  const trainingSchemaVersion = assertTrainingSchemaVersion(options.trainingSchemaVersion);
  const startTime = performance.now();
  await trainingHarness.getGPU();

  const benchSettings = resolveBenchRunSettings(options);
  const totalRuns = benchSettings.warmupRuns + benchSettings.timedRuns;
  const trainingOverrides = resolveTrainingOverrides(options);
  const adapterActivation = normalizeAdapterActivationConfig(options);
  const distillEnabled = trainingOverrides?.distill?.enabled === true;
  const distillDatasetPath = resolveDistillDatasetPath(options, trainingOverrides);
  const distillDatasetReport = distillEnabled
    ? await loadDistillDatasetFromJsonl(distillDatasetPath)
    : null;
  const resolvedResumeFrom = options.resumeFrom || trainingOverrides?.distill?.resumeFrom || null;
  const resolvedStage1Artifact = options.stage1Artifact || trainingOverrides?.ul?.stage1Artifact || null;
  const resolvedStage1ArtifactHash = options.stage1ArtifactHash || trainingOverrides?.ul?.stage1ArtifactHash || null;
  const resolvedStageAArtifact = options.stageAArtifact || trainingOverrides?.distill?.stageAArtifact || null;
  const resolvedStageAArtifactHash = options.stageAArtifactHash || trainingOverrides?.distill?.stageAArtifactHash || null;
  let distillRuntime = null;
  if (distillEnabled) {
    if (!distillDatasetPath) {
      throw new Error('Distill benchmark requires --distill-dataset-path (training.distill.datasetPath).');
    }
    distillRuntime = await createDistillRuntimeContext(options, trainingOverrides);
  }

  const timedRunDurationsMs = [];
  const timedRunStepsPerSec = [];
  const timedStepDurationsMs = [];
  const timedRunUlArtifacts = [];
  const timedRunDistillArtifacts = [];
  const timedRunAdapterExports = [];
  const trainingMetricsReport = [];
  const distillShardProgress = resolveDistillShardProgressContext(
    options,
    trainingOverrides,
    benchSettings.stepsPerRun,
    distillDatasetReport?.shardCount ?? null
  );
  const checkpointResumeTimeline = [];
  appendTimelineEvent(checkpointResumeTimeline, 'benchmark_started', {
    workloadType: 'training',
    trainingStage: (
      options.trainingStage
      || trainingOverrides?.distill?.stage
      || trainingOverrides?.ul?.stage
      || null
    ),
    shardIndex: distillShardProgress.shardIndex,
    shardCount: distillShardProgress.shardCount,
    stepsPerShard: distillShardProgress.stepsPerShard,
  });
  if (resolvedResumeFrom) {
    appendTimelineEvent(checkpointResumeTimeline, 'resume_requested', {
      resumeFrom: String(resolvedResumeFrom),
    });
  }
  if (resolvedStage1Artifact) {
    appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_declared', {
      dependencyType: 'ul_stage1',
      stage1Artifact: String(resolvedStage1Artifact),
      stage1ArtifactHash: resolvedStage1ArtifactHash,
    });
  }
  if (resolvedStageAArtifact) {
    appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_declared', {
      dependencyType: 'distill_stage_a',
      stageAArtifact: String(resolvedStageAArtifact),
      stageAArtifactHash: resolvedStageAArtifactHash,
    });
  }
  let completedTimedRuns = 0;
  let latestExportedAdapter = null;

  try {
    for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
      const fixture = distillEnabled
        ? await createDistillStudentRuntimeModelFixture({
          training: trainingOverrides,
        }, {
          outputDim: distillRuntime?.topK ?? DISTILL_ADAPTER_TOP_K,
          distillRuntime,
        })
        : createToyModelFixture({
          training: trainingOverrides,
        });
      try {
        const runner = new TrainingRunner(fixture.config, {
          optimizer: new AdamOptimizer(fixture.config),
          crossEntropyLoss,
          clipGradients,
        });
        const dataset = distillEnabled
          ? distillDatasetReport.createDataset({
            batchSize: 1,
            shuffle: false,
            seed: 1337 + runIndex,
            distillRuntime,
          })
          : {
            async *batches() {
              for (let i = 0; i < benchSettings.stepsPerRun; i += 1) {
                yield fixture.batch;
              }
            },
          };

        const runStart = performance.now();
        const isTimedRun = runIndex >= benchSettings.warmupRuns;
        appendTimelineEvent(checkpointResumeTimeline, 'run_started', {
          runIndex: runIndex + 1,
          phase: isTimedRun ? 'timed' : 'warmup',
        });
        const runMetrics = await runner.run(fixture.model, dataset, {
          epochs: 1,
          batchSize: 1,
          shuffle: false,
          maxSteps: benchSettings.stepsPerRun,
          modelId: options.modelId || distillRuntime?.studentModelId || 'training',
          modelUrl: options.modelUrl || distillRuntime?.studentModelUrl || null,
          timestamp: options.timestamp || null,
          ulArtifactDir: options.ulArtifactDir || null,
          distillArtifactDir: options.distillArtifactDir || null,
          stageAArtifact: resolvedStageAArtifact,
          stageAArtifactHash: resolvedStageAArtifactHash,
          teacherModelId: distillRuntime?.teacherModelId || options.teacherModelId || null,
          studentModelId: distillRuntime?.studentModelId || options.studentModelId || null,
          distillDatasetId: options.distillDatasetId || null,
          distillDatasetPath: distillDatasetReport?.absolutePath || null,
          distillLanguagePair: options.distillLanguagePair || null,
          distillShardIndex: distillShardProgress.shardIndex,
          distillShardCount: distillShardProgress.shardCount,
          resumeFrom: resolvedResumeFrom,
        });
        const runDurationMs = Math.max(0, performance.now() - runStart);
        if (runner.resumeState && typeof runner.resumeState === 'object') {
          appendTimelineEvent(checkpointResumeTimeline, 'run_resumed', {
            runIndex: runIndex + 1,
            phase: isTimedRun ? 'timed' : 'warmup',
            resumedStep: runner.resumeState.step ?? null,
            resumedEpoch: runner.resumeState.epoch ?? null,
            resumedBatch: runner.resumeState.batch ?? null,
            resumedCheckpointHash: runner.resumeState.checkpointHash ?? null,
          });
        }
        appendTimelineEvent(checkpointResumeTimeline, 'run_completed', {
          runIndex: runIndex + 1,
          phase: isTimedRun ? 'timed' : 'warmup',
          durationMs: runDurationMs,
          stepCount: Array.isArray(runMetrics) ? runMetrics.length : 0,
        });
        if (isTimedRun) {
          completedTimedRuns += 1;
          timedRunDurationsMs.push(runDurationMs);
          const runStepCount = Array.isArray(runMetrics) ? runMetrics.length : 0;
          if (runDurationMs > 0 && runStepCount > 0) {
            timedRunStepsPerSec.push((runStepCount * 1000) / runDurationMs);
          }
          for (const stepEntry of runMetrics) {
            const stepWithRun = {
              ...stepEntry,
              bench_run_index: completedTimedRuns,
              bench_run_global_index: runIndex + 1,
            };
            if (isFiniteNumber(stepWithRun?.step_time_ms)) {
              timedStepDurationsMs.push(stepWithRun.step_time_ms);
            }
            trainingMetricsReport.push(stepWithRun);
          }
          if (runner.lastCheckpoint && typeof runner.lastCheckpoint === 'object') {
            appendTimelineEvent(checkpointResumeTimeline, 'checkpoint_state_written', {
              runIndex: runIndex + 1,
              timedRunIndex: completedTimedRuns,
              checkpointKey: runner.lastCheckpoint.key || null,
              checkpointStep: runner.lastCheckpoint.step ?? null,
              checkpointEpoch: runner.lastCheckpoint.epoch ?? null,
              checkpointBatch: runner.lastCheckpoint.batch ?? null,
            });
          }
          if (runner.lastArtifact && typeof runner.lastArtifact === 'object') {
            const artifactEntry = {
              runIndex: completedTimedRuns,
              ...runner.lastArtifact,
            };
            appendTimelineEvent(checkpointResumeTimeline, 'checkpoint_written', {
              runIndex: runIndex + 1,
              timedRunIndex: completedTimedRuns,
              artifactKind: artifactEntry.kind || null,
              stage: artifactEntry.stage || null,
              manifestPath: artifactEntry.manifestPath || null,
              manifestHash: artifactEntry.manifestHash || null,
              manifestFileHash: artifactEntry.manifestFileHash || null,
            });
            if (artifactEntry.stageADependency) {
              appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_resolved', {
                dependencyType: 'distill_stage_a',
                runIndex: runIndex + 1,
                stageADependency: artifactEntry.stageADependency,
              });
            }
            if (artifactEntry.stage1Dependency) {
              appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_resolved', {
                dependencyType: 'ul_stage1',
                runIndex: runIndex + 1,
                stage1Dependency: artifactEntry.stage1Dependency,
              });
            }
            if (runner.lastArtifact.kind === 'distill') {
              timedRunDistillArtifacts.push(artifactEntry);
            } else {
              timedRunUlArtifacts.push(artifactEntry);
            }
          }
          if (adapterActivation.enabled && adapterActivation.exportConfig) {
            const exportedAdapter = await exportLoRAAdapterFromModel(
              fixture.model,
              adapterActivation.exportConfig,
              completedTimedRuns
            );
            if (exportedAdapter) {
              latestExportedAdapter = exportedAdapter;
              timedRunAdapterExports.push({
                runIndex: completedTimedRuns,
                id: exportedAdapter.manifest?.id || null,
                name: exportedAdapter.manifest?.name || null,
                hash: exportedAdapter.hash,
              });
            }
          }
        }
      } finally {
        fixture.cleanup();
      }
    }
  } finally {
    if (distillRuntime && typeof distillRuntime.cleanup === 'function') {
      await distillRuntime.cleanup();
    }
  }

  const runMsStats = computeSampleStats(timedRunDurationsMs);
  const stepMsStats = computeSampleStats(timedStepDurationsMs);
  const stepsPerSecStats = computeSampleStats(timedRunStepsPerSec);
  const progress = resolveBenchProgressSummary(trainingMetricsReport, distillShardProgress, startTime);
  const activationPayload = adapterActivation.adapterPayload
    ? adapterActivation.adapterPayload
    : (latestExportedAdapter
      ? {
        adapterManifest: latestExportedAdapter.manifest,
        adapterManifestJson: latestExportedAdapter.json,
      }
      : null);
  const adapterActivationResult = (
    adapterActivation.enabled
    && adapterActivation.autoActivate
  )
    ? await tryActivateAdapterPayload(activationPayload)
    : null;
  appendTimelineEvent(checkpointResumeTimeline, 'benchmark_completed', {
    completedTimedRuns,
    metricEntryCount: trainingMetricsReport.length,
    percentComplete: progress.percentComplete,
    etaMs: progress.etaMs,
  });

  const results = [
    {
      name: 'training-benchmark',
      passed: completedTimedRuns > 0 && trainingMetricsReport.length > 0,
      duration: Math.max(0, performance.now() - startTime),
      error: completedTimedRuns > 0 && trainingMetricsReport.length > 0
        ? undefined
        : 'No timed training benchmark runs completed.',
    },
  ];

  const summary = buildSuiteSummary('bench', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || distillRuntime?.studentModelId || options.modelUrl || 'training',
    metrics: {
      workloadType: 'training',
      warmupRuns: benchSettings.warmupRuns,
      timedRuns: benchSettings.timedRuns,
      completedTimedRuns,
      stepsPerRun: benchSettings.stepsPerRun,
      trainingSchemaVersion,
      trainingMetricsReport,
      progress,
      ulArtifacts: timedRunUlArtifacts,
      distillArtifacts: timedRunDistillArtifacts,
      adapterExports: timedRunAdapterExports,
      adapterActivation: adapterActivationResult,
      checkpointResumeTimeline,
      distillDataset: distillDatasetReport
        ? {
          path: distillDatasetReport.absolutePath,
          rowCount: distillDatasetReport.rowCount,
          sampleCount: distillDatasetReport.sampleCount,
          shardCount: distillDatasetReport.shardCount ?? 1,
          directionCounts: distillDatasetReport.directionCounts,
        }
        : null,
      latency: {
        runMs: runMsStats,
        stepMs: stepMsStats,
      },
      throughput: {
        stepsPerSec: stepsPerSecStats,
      },
    },
    deviceInfo: getKernelCapabilities(),
  };
}
