import { initDevice, getKernelCapabilities, getDevice } from '../gpu/device.js';
import { setPlatformsBaseUrl } from '../config/platforms/loader.js';
import { setRegistryUrl } from '../config/kernels/registry.js';
import { createTrainingConfig } from '../config/training-defaults.js';
import { runMatmul } from '../gpu/kernels/index.js';
import { createTensor } from '../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../memory/buffer-pool.js';
import { OpType } from './autograd.js';
import { AdamOptimizer } from './optimizer.js';
import { TrainingRunner } from './runner.js';
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
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
const DISTILL_EPS = 1e-8;

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

function klDivergence(teacherProbs, studentProbs) {
  const size = Math.min(teacherProbs.length, studentProbs.length);
  if (size <= 0) return 0;
  let total = 0;
  for (let i = 0; i < size; i += 1) {
    const p = Math.max(DISTILL_EPS, teacherProbs[i]);
    const q = Math.max(DISTILL_EPS, studentProbs[i]);
    total += p * (Math.log(p) - Math.log(q));
  }
  return total;
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

async function computePromptDistillFeatures(sample, prompt, runtime) {
  const teacherResult = await runtime.teacherPipeline.prefillWithLogits(prompt, {
    useChatTemplate: false,
  });
  const studentResult = await runtime.studentPipeline.prefillWithLogits(prompt, {
    useChatTemplate: false,
  });

  try {
    const teacherLogits = toFloat32Array(teacherResult?.logits, 'teacher logits');
    const studentLogits = toFloat32Array(studentResult?.logits, 'student logits');
    const topTokenIndices = selectTopKIndices(teacherLogits, runtime.topK);
    const teacherTopLogits = gatherLogitsByIndices(teacherLogits, topTokenIndices, DISTILL_LOGIT_FALLBACK);
    const studentTopLogits = gatherLogitsByIndices(studentLogits, topTokenIndices, DISTILL_LOGIT_FALLBACK);
    const teacherTopProbs = softmax(teacherTopLogits, runtime.temperature);
    const studentTopProbs = softmax(studentTopLogits, runtime.temperature);
    const targetClass = argmax(teacherTopLogits);
    return {
      source: sample.source,
      direction: sample.direction,
      targetClass,
      topTokenIndices: Array.from(topTokenIndices),
      teacherTopLogits,
      studentTopLogits,
      teacherTopProbs,
      kdLoss: klDivergence(teacherTopProbs, studentTopProbs),
    };
  } finally {
    disposePrefillSnapshot(teacherResult);
    disposePrefillSnapshot(studentResult);
    runtime.teacherPipeline.reset();
    runtime.studentPipeline.reset();
  }
}

function createDistillTensorDataset(samples, options = {}) {
  if (!Array.isArray(samples) || samples.length === 0) {
    throw new Error('Distill dataset has no usable rows.');
  }
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  if (!distillRuntime?.teacherPipeline || !distillRuntime?.studentPipeline) {
    throw new Error('Distill dataset requires teacherPipeline and studentPipeline.');
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
          const studentTopLogits = [];
          const teacherTargetIndices = [];
          const tripletLossValues = [];
          let kdLossSum = 0;
          let tripletLossSum = 0;
          let tripletLossCount = 0;
          const directionCounts = {};

          for (let i = 0; i < batchIndices.length; i += 1) {
            const sample = samples[batchIndices[i]];
            const prompt = buildDistillPrompt(sample);
            const baseDistill = await computePromptDistillFeatures(sample, prompt, {
              ...distillRuntime,
              topK,
            });

            const baseOffset = i * topK;
            features.set(baseDistill.studentTopLogits, baseOffset);
            targets[i] = baseDistill.targetClass;
            teacherTargetIndices.push(baseDistill.targetClass);
            teacherTopProbs.push(baseDistill.teacherTopProbs);
            teacherTopTokenIndices.push(baseDistill.topTokenIndices);
            teacherTopLogits.push(baseDistill.teacherTopLogits);
            studentTopLogits.push(baseDistill.studentTopLogits);
            kdLossSum += baseDistill.kdLoss;

            if (stage === 'stage_b' && sample.targetNeg) {
              const posPrompt = buildDistillCandidatePrompt(sample, sample.targetPos);
              const negPrompt = buildDistillCandidatePrompt(sample, sample.targetNeg);
              const positive = await computePromptDistillFeatures(sample, posPrompt, {
                ...distillRuntime,
                topK,
              });
              const negative = await computePromptDistillFeatures(sample, negPrompt, {
                ...distillRuntime,
                topK,
              });
              const tripletLoss = positive.kdLoss - negative.kdLoss;
              tripletLossValues.push(tripletLoss);
              tripletLossSum += tripletLoss;
              tripletLossCount += 1;
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
              teacherTopProbs,
              teacherTopTokenIndices,
              teacherTopLogits,
              studentTopLogits,
              teacherTargetIndices,
              kdLossMean: batchIndices.length > 0 ? (kdLossSum / batchIndices.length) : 0,
              tripletLossValues,
              tripletLossMean: tripletLossCount > 0 ? (tripletLossSum / tripletLossCount) : 0,
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

  const [{ readFile }, { resolve }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
  ]);

  const absolutePath = resolve(normalizedPath);
  let raw;
  try {
    raw = await readFile(absolutePath, 'utf8');
  } catch (error) {
    const message = error?.message ? String(error.message) : String(error);
    throw new Error(`Failed to read distillDatasetPath "${absolutePath}": ${message}`);
  }

  const rows = parseJsonl(raw);
  const encodedRows = [];
  for (let i = 0; i < rows.length; i += 1) {
    const encoded = encodeDistillRow(rows[i], i);
    if (encoded) encodedRows.push(encoded);
  }
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

async function loadDistillModelHandle(modelRef, role) {
  const normalizedRef = normalizeOptionalString(modelRef);
  if (!normalizedRef) {
    throw new Error(`Distill ${role} model reference is required.`);
  }

  const loadFromUrl = async (url) => {
    const initialized = await initializeInference(url, {
      log: () => {},
      onProgress: () => {},
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
  const teacher = await loadDistillModelHandle(teacherModelRef, 'teacher');
  let student = null;
  try {
    student = await loadDistillModelHandle(studentModelRef, 'student');
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

function createDistillAdapterModelFixture(overrides = {}, options = {}) {
  const inputDim = clampDistillTopK(options.inputDim ?? DISTILL_ADAPTER_TOP_K);
  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const adapterWeights = new Float32Array(inputDim * inputDim);
  for (let row = 0; row < inputDim; row += 1) {
    const offset = row * inputDim;
    adapterWeights[offset + row] = 1;
  }
  const adapterWeight = makeTensorFromFloat32(
    adapterWeights,
    [inputDim, inputDim],
    'distill_adapter_weight'
  );

  const model = {
    async forward(inputTensor, tape) {
      const rows = Number.isFinite(inputTensor?.shape?.[0]) ? inputTensor.shape[0] : 1;
      return tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, rows, inputDim, inputDim, { transposeB: false }),
        [inputTensor, adapterWeight],
        { M: rows, N: inputDim, K: inputDim, transposeB: false }
      );
    },
    loraParams() {
      return [adapterWeight];
    },
    paramGroups() {
      return {
        encoder: [],
        prior: [],
        decoder: [],
        base: [adapterWeight],
        lora: [adapterWeight],
      };
    },
  };

  return {
    config,
    model,
    inputDim,
    cleanup() {
      releaseTensor(adapterWeight);
    },
  };
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
    artifactDir: options.ulArtifactDir ?? trainingConfig?.ul?.artifactDir ?? 'bench/out/ul',
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
    stageAArtifact: options.stageAArtifact ?? trainingConfig?.distill?.stageAArtifact ?? null,
    stageAArtifactHash: options.stageAArtifactHash ?? trainingConfig?.distill?.stageAArtifactHash ?? null,
    artifactDir: options.distillArtifactDir ?? trainingConfig?.distill?.artifactDir ?? 'bench/out/distill',
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
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 2,
      modelId: options.modelId || 'training',
      modelUrl: options.modelUrl || null,
      timestamp: options.timestamp || null,
      ulArtifactDir: options.ulArtifactDir || null,
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
  const adapterInputDim = clampDistillTopK(distillTraining?.distill?.topK ?? DISTILL_ADAPTER_TOP_K);
  const fixture = createDistillAdapterModelFixture({
    training: distillTraining || undefined,
  }, {
    inputDim: adapterInputDim,
  });
  let distillRuntime = null;

  try {
    const distillDatasetPath = resolveDistillDatasetPath(options, fixture.config.training);
    if (!distillDatasetPath) {
      throw new Error('Distill stage requires --distill-dataset-path (training.distill.datasetPath).');
    }
    const distillDatasetReport = await loadDistillDatasetFromJsonl(distillDatasetPath);
    distillRuntime = await createDistillRuntimeContext({
      ...options,
      trainingStage: stage,
    }, fixture.config.training);

    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
    });
    const dataset = distillDatasetReport.createDataset({
      batchSize: 1,
      shuffle: false,
      seed: 1337,
      distillRuntime,
    });
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: 1,
      shuffle: false,
      maxSteps: 2,
      modelId: options.modelId || distillRuntime.studentModelId || 'training',
      modelUrl: options.modelUrl || distillRuntime.studentModelUrl || null,
      timestamp: options.timestamp || null,
      distillArtifactDir: options.distillArtifactDir || null,
      stageAArtifact: options.stageAArtifact || null,
      stageAArtifactHash: options.stageAArtifactHash || null,
      teacherModelId: distillRuntime.teacherModelId || null,
      studentModelId: distillRuntime.studentModelId || null,
      distillDatasetId: options.distillDatasetId || null,
      distillDatasetPath: distillDatasetReport.absolutePath,
      distillLanguagePair: options.distillLanguagePair || null,
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
        distillResolvedConfig: {
          enabled: fixture.config.training?.distill?.enabled === true,
          stage: fixture.config.training?.distill?.stage ?? null,
          teacherModelId: fixture.config.training?.distill?.teacherModelId ?? null,
          studentModelId: fixture.config.training?.distill?.studentModelId ?? null,
          datasetId: fixture.config.training?.distill?.datasetId ?? null,
          datasetPath: fixture.config.training?.distill?.datasetPath ?? null,
          languagePair: fixture.config.training?.distill?.languagePair ?? null,
          temperature: fixture.config.training?.distill?.temperature ?? null,
          alphaKd: fixture.config.training?.distill?.alphaKd ?? null,
          alphaCe: fixture.config.training?.distill?.alphaCe ?? null,
          tripletMargin: fixture.config.training?.distill?.tripletMargin ?? null,
          topK: fixture.config.training?.distill?.topK ?? adapterInputDim,
          freeze: fixture.config.training?.distill?.freeze ?? null,
        },
        distillRuntime: {
          teacherModelId: distillRuntime.teacherModelId || null,
          studentModelId: distillRuntime.studentModelId || null,
          teacherModelUrl: distillRuntime.teacherModelUrl || null,
          studentModelUrl: distillRuntime.studentModelUrl || null,
          topK: distillRuntime.topK,
        },
        distillDataset: {
          path: distillDatasetReport.absolutePath,
          rowCount: distillDatasetReport.rowCount,
          sampleCount: distillDatasetReport.sampleCount,
          directionCounts: distillDatasetReport.directionCounts,
        },
      },
    };
  } finally {
    if (distillRuntime && typeof distillRuntime.cleanup === 'function') {
      await distillRuntime.cleanup();
    }
    fixture.cleanup();
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
  return {
    ...summary,
    modelId: options.modelId || options.modelUrl || 'training',
    metrics: {
      testsRun: results.length,
      selectedTests: testsToRun,
      availableTests,
      trainingStage: requestedStage || null,
      trainingSchemaVersion,
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
  const distillEnabled = trainingOverrides?.distill?.enabled === true;
  const distillDatasetPath = resolveDistillDatasetPath(options, trainingOverrides);
  const distillDatasetReport = distillEnabled
    ? await loadDistillDatasetFromJsonl(distillDatasetPath)
    : null;
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
  const trainingMetricsReport = [];
  let completedTimedRuns = 0;

  try {
    for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
      const fixture = distillEnabled
        ? createDistillAdapterModelFixture({
          training: trainingOverrides,
        }, {
          inputDim: distillRuntime?.topK ?? DISTILL_ADAPTER_TOP_K,
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
          stageAArtifact: options.stageAArtifact || null,
          stageAArtifactHash: options.stageAArtifactHash || null,
          teacherModelId: distillRuntime?.teacherModelId || options.teacherModelId || null,
          studentModelId: distillRuntime?.studentModelId || options.studentModelId || null,
          distillDatasetId: options.distillDatasetId || null,
          distillDatasetPath: distillDatasetReport?.absolutePath || null,
          distillLanguagePair: options.distillLanguagePair || null,
        });
        const runDurationMs = Math.max(0, performance.now() - runStart);
        const isTimedRun = runIndex >= benchSettings.warmupRuns;
        if (isTimedRun) {
          completedTimedRuns += 1;
          timedRunDurationsMs.push(runDurationMs);
          const runStepCount = Array.isArray(runMetrics) ? runMetrics.length : 0;
          if (runDurationMs > 0 && runStepCount > 0) {
            timedRunStepsPerSec.push((runStepCount * 1000) / runDurationMs);
          }
          for (const stepEntry of runMetrics) {
            if (isFiniteNumber(stepEntry?.step_time_ms)) {
              timedStepDurationsMs.push(stepEntry.step_time_ms);
            }
            trainingMetricsReport.push(stepEntry);
          }
          if (runner.lastArtifact && typeof runner.lastArtifact === 'object') {
            const artifactEntry = {
              runIndex: completedTimedRuns,
              ...runner.lastArtifact,
            };
            if (runner.lastArtifact.kind === 'distill') {
              timedRunDistillArtifacts.push(artifactEntry);
            } else {
              timedRunUlArtifacts.push(artifactEntry);
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
      ulArtifacts: timedRunUlArtifacts,
      distillArtifacts: timedRunDistillArtifacts,
      distillDataset: distillDatasetReport
        ? {
          path: distillDatasetReport.absolutePath,
          rowCount: distillDatasetReport.rowCount,
          sampleCount: distillDatasetReport.sampleCount,
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
