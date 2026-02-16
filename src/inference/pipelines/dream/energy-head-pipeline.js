import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { createTensor, dtypeBytes } from '../../../gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { runEnergyEval, runEnergyUpdate } from '../../../gpu/kernels/index.js';
import { log } from '../../../debug/index.js';
import { f16ToF32Array, f32ToF16Array } from '../../kv-cache/types.js';
import { registerPipeline } from '../registry.js';
import { applyPipelineContexts } from '../context.js';
import { createInitializedPipeline } from '../factory.js';

const DREAM_ENERGY_HEAD_MODEL_TYPES = Object.freeze([
  'dream_energy_head',
  'dream-energy-head',
  'd1-to2-bridge-diffusion',
  'synthesis-mixer-diffusion',
  'ebrm-diffusion',
]);

const DEFAULT_INFER_CONFIG = Object.freeze({
  backend: 'auto',
  dtype: 'f32',
  steps: 3,
  stepSize: 0.08,
  gradientScale: 1,
  energyScale: 1,
});

function isObj(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function toFinite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, toFinite(value, 0)));
}

function sigmoid(value) {
  if (value >= 30) return 1;
  if (value <= -30) return 0;
  return 1 / (1 + Math.exp(-value));
}

function dot(a, b) {
  let out = 0;
  for (let i = 0; i < a.length; i++) {
    out += a[i] * b[i];
  }
  return out;
}

function normalizeMainHead(model) {
  const featureIds = Array.isArray(model?.featureIds) ? model.featureIds.map(String) : [];
  const weights = Array.isArray(model?.weights) ? model.weights.map((value) => toFinite(value, 0)) : [];
  if (!featureIds.length || !weights.length || weights.length !== featureIds.length) {
    throw new Error('DreamEnergyHeadPipeline.loadModel: featureIds/weights mismatch.');
  }
  return {
    featureIds,
    weights: new Float32Array(weights),
    bias: toFinite(model?.bias, 0),
    scale: 1,
  };
}

function normalizeAuxHead(head) {
  const featureIds = Array.isArray(head?.featureIds) ? head.featureIds.map(String) : [];
  const weights = Array.isArray(head?.weights) ? head.weights.map((value) => toFinite(value, 0)) : [];
  if (!featureIds.length || !weights.length || weights.length !== featureIds.length) {
    return null;
  }
  return {
    featureIds,
    weights: new Float32Array(weights),
    bias: toFinite(head?.bias, 0),
    scale: Number.isFinite(head?.scale) ? Number(head.scale) : 1,
  };
}

function normalizeModelManifest(manifest) {
  const modelType = String(manifest?.modelType || '');
  if (!DREAM_ENERGY_HEAD_MODEL_TYPES.includes(modelType)) {
    throw new Error(`DreamEnergyHeadPipeline.loadModel: unsupported modelType "${modelType}".`);
  }

  const mainHead = normalizeMainHead(manifest);
  const localHead = normalizeAuxHead(manifest?.localHead);
  const treeHead = normalizeAuxHead(manifest?.treeHead);
  const consistencyHead = normalizeAuxHead(manifest?.consistencyHead);
  return {
    modelType,
    modelId: String(manifest?.modelId || 'dream-energy-head'),
    modelHash: manifest?.modelHash || null,
    mainHead,
    localHead,
    treeHead,
    consistencyHead,
  };
}

function resolveHead(model, headId) {
  const id = String(headId || 'main');
  if (id === 'main') return model.mainHead;
  if (id === 'local') {
    if (!model.localHead) throw new Error('DreamEnergyHeadPipeline: local head unavailable in model.');
    return model.localHead;
  }
  if (id === 'tree') {
    if (!model.treeHead) throw new Error('DreamEnergyHeadPipeline: tree head unavailable in model.');
    return model.treeHead;
  }
  if (id === 'consistency') {
    if (!model.consistencyHead) {
      throw new Error('DreamEnergyHeadPipeline: consistency head unavailable in model.');
    }
    return model.consistencyHead;
  }
  throw new Error(`DreamEnergyHeadPipeline: unsupported head "${id}".`);
}

function resolveActivation({ modelType, headId, activation }) {
  if (typeof activation === 'string' && activation) return activation;
  if (String(modelType).includes('ebrm') && ['main', 'local', 'tree', 'consistency'].includes(String(headId || 'main'))) {
    return 'linear';
  }
  return 'sigmoid';
}

function alignFeatureVector(row, head) {
  if (Array.isArray(row?.features)) {
    if (row.features.length !== head.featureIds.length) {
      throw new Error('DreamEnergyHeadPipeline.scoreRows: features length mismatch.');
    }
    return new Float32Array(row.features.map((value) => toFinite(value, 0)));
  }
  if (isObj(row?.features)) {
    return new Float32Array(
      head.featureIds.map((id) => toFinite(row.features[id], 0))
    );
  }
  throw new Error('DreamEnergyHeadPipeline.scoreRows: row.features must be array or object.');
}

function applyActivation(logit, activation) {
  if (activation === 'linear') return logit;
  if (activation === 'sigmoid') return sigmoid(logit);
  throw new Error(`DreamEnergyHeadPipeline: unsupported activation "${activation}".`);
}

function resolveBackend(backend) {
  const mode = String(backend || DEFAULT_INFER_CONFIG.backend).toLowerCase();
  if (mode === 'gpu' || mode === 'cpu') return mode;
  const device = getDevice();
  return device ? 'gpu' : 'cpu';
}

async function createFeatureTensor(device, values, dtype, label) {
  const payload = dtype === 'f16' ? f32ToF16Array(values) : values;
  const byteLength = payload.byteLength;
  const alignedSize = Math.ceil(byteLength / 4) * 4;
  const buffer = acquireBuffer(alignedSize, undefined, label);

  if (alignedSize === byteLength) {
    device.queue.writeBuffer(buffer, 0, payload);
  } else {
    const bytes = payload instanceof Uint16Array
      ? new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength)
      : new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
    const padded = new Uint8Array(alignedSize);
    padded.set(bytes);
    device.queue.writeBuffer(buffer, 0, padded);
  }

  return createTensor(buffer, dtype, [values.length], label);
}

async function readTensorF32(tensor) {
  const byteLength = tensor.shape[0] * dtypeBytes(tensor.dtype);
  const data = await readBuffer(tensor.buffer, byteLength);
  if (tensor.dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(data));
  }
  return new Float32Array(data);
}

function scoreRowCpu({
  features,
  head,
  steps,
  stepSize,
  gradientScale,
  energyScale,
  activation,
}) {
  const state = new Float32Array(features);
  for (let step = 0; step < steps; step++) {
    for (let i = 0; i < state.length; i++) {
      const grad = (head.weights[i] - state[i]) * gradientScale;
      state[i] += stepSize * grad;
    }
  }

  let energy = 0;
  for (let i = 0; i < state.length; i++) {
    const delta = state[i] - head.weights[i];
    energy += delta * delta;
  }
  energy /= state.length || 1;

  const raw = dot(head.weights, state) + head.bias;
  const logit = raw - (energyScale * energy);
  const score = applyActivation(logit, activation) * head.scale;
  return { score, logit, energy };
}

async function scoreRowGpu({
  features,
  head,
  steps,
  stepSize,
  gradientScale,
  energyScale,
  activation,
  dtype,
  targetTensor = null,
  releaseTargetTensor = true,
}) {
  const device = getDevice();
  if (!device) {
    throw new Error('DreamEnergyHeadPipeline: GPU backend requested but no WebGPU device is available.');
  }

  let stateTensor = null;
  let activeTargetTensor = targetTensor;
  let energyTensor = null;
  try {
    stateTensor = await createFeatureTensor(device, features, dtype, 'dream_head_state');
    if (!activeTargetTensor) {
      activeTargetTensor = await createFeatureTensor(device, head.weights, dtype, 'dream_head_target');
    }

    for (let step = 0; step < steps; step++) {
      await runEnergyUpdate(stateTensor, activeTargetTensor, {
        count: features.length,
        stepSize,
        gradientScale,
      });
    }

    energyTensor = await runEnergyEval(stateTensor, activeTargetTensor, {
      count: features.length,
      scale: 1,
    });
    const energyValues = await readTensorF32(energyTensor);
    let energy = 0;
    for (let i = 0; i < energyValues.length; i++) {
      energy += energyValues[i];
    }
    energy /= energyValues.length || 1;

    const stateValues = await readTensorF32(stateTensor);
    const raw = dot(head.weights, stateValues) + head.bias;
    const logit = raw - (energyScale * energy);
    const score = applyActivation(logit, activation) * head.scale;
    return { score, logit, energy };
  } finally {
    if (stateTensor?.buffer) releaseBuffer(stateTensor.buffer);
    if (releaseTargetTensor && activeTargetTensor?.buffer) releaseBuffer(activeTargetTensor.buffer);
    if (energyTensor?.buffer) releaseBuffer(energyTensor.buffer);
  }
}

export class DreamEnergyHeadPipeline {
  runtimeConfig = null;
  manifest = null;
  model = null;
  stats = {};
  baseUrl = null;
  _onProgress = null;

  async initialize(contexts = {}) {
    const { runtimeConfig } = applyPipelineContexts(this, contexts);
    this.runtimeConfig = runtimeConfig;
  }

  async loadModel(manifest) {
    this.manifest = manifest || null;
    this.model = normalizeModelManifest(manifest);
    log.info('DreamEnergyHead', `Loaded model "${this.model.modelId}" (${this.model.modelType})`);
  }

  getStats() {
    return this.stats;
  }

  getMemoryStats() {
    return { used: 0, kvCache: null };
  }

  async unload() {
    this.manifest = null;
    this.model = null;
    this.stats = {};
  }

  async scoreRows(request = {}) {
    if (!this.model) {
      throw new Error('DreamEnergyHeadPipeline.scoreRows: model is not loaded.');
    }
    const rows = Array.isArray(request?.rows) ? request.rows : [];
    if (!rows.length) {
      throw new Error('DreamEnergyHeadPipeline.scoreRows: rows[] is required.');
    }

    const headId = String(request?.head || 'main');
    const head = resolveHead(this.model, headId);
    const activation = resolveActivation({
      modelType: this.model.modelType,
      headId,
      activation: request?.activation,
    });
    const backend = resolveBackend(request?.backend);
    const steps = Math.max(0, Math.floor(toFinite(request?.steps, DEFAULT_INFER_CONFIG.steps)));
    const stepSize = toFinite(request?.stepSize, DEFAULT_INFER_CONFIG.stepSize);
    const gradientScale = toFinite(request?.gradientScale, DEFAULT_INFER_CONFIG.gradientScale);
    const energyScale = toFinite(request?.energyScale, DEFAULT_INFER_CONFIG.energyScale);

    let caps = { hasF16: false };
    try {
      caps = getKernelCapabilities();
    } catch {}
    const requestedDtype = String(request?.dtype || DEFAULT_INFER_CONFIG.dtype);
    const dtype = requestedDtype === 'f16' && caps.hasF16 ? 'f16' : 'f32';

    const startTime = performance.now();
    const outputRows = [];
    let usedBackend = backend;
    let autoBackendFailure = false;
    let sharedGpuTargetTensor = null;
    try {
      if (backend === 'gpu' || backend === 'auto') {
        const device = getDevice();
        if (device) {
          sharedGpuTargetTensor = await createFeatureTensor(device, head.weights, dtype, 'dream_head_target_shared');
        }
      }

      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const rowId = String(row?.rowId ?? row?.candidateId ?? i);
        const features = alignFeatureVector(row, head);

        let scored = null;
        if (backend === 'gpu') {
          scored = await scoreRowGpu({
            features,
            head,
            steps,
            stepSize,
            gradientScale,
            energyScale,
            activation,
            dtype,
            targetTensor: sharedGpuTargetTensor,
            releaseTargetTensor: sharedGpuTargetTensor == null,
          });
        } else if (backend === 'cpu') {
          scored = scoreRowCpu({
            features,
            head,
            steps,
            stepSize,
            gradientScale,
            energyScale,
            activation,
          });
        } else if (autoBackendFailure) {
          scored = scoreRowCpu({
            features,
            head,
            steps,
            stepSize,
            gradientScale,
            energyScale,
            activation,
          });
        } else {
          try {
            scored = await scoreRowGpu({
              features,
              head,
              steps,
              stepSize,
              gradientScale,
              energyScale,
              activation,
              dtype,
              targetTensor: sharedGpuTargetTensor,
              releaseTargetTensor: sharedGpuTargetTensor == null,
            });
            usedBackend = 'gpu';
          } catch (error) {
            log.warn('DreamEnergyHead', `GPU score fallback to CPU: ${error?.message || error}`);
            scored = scoreRowCpu({
              features,
              head,
              steps,
              stepSize,
              gradientScale,
              energyScale,
              activation,
            });
            usedBackend = 'cpu';
            autoBackendFailure = true;
          }
        }

        outputRows.push({
          rowId,
          score: activation === 'sigmoid'
            ? clamp01(scored.score)
            : toFinite(scored.score, 0),
          logit: toFinite(scored.logit, 0),
          energy: Math.max(0, toFinite(scored.energy, 0)),
        });

        if (this._onProgress) {
          this._onProgress({
            stage: 'dream_energy_head',
            percent: (i + 1) / rows.length,
            message: `Scored ${i + 1} / ${rows.length}`,
          });
        }
      }
    } finally {
      if (sharedGpuTargetTensor?.buffer) releaseBuffer(sharedGpuTargetTensor.buffer);
    }

    outputRows.sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score;
      return a.rowId.localeCompare(b.rowId);
    });

    this.stats = {
      backend: usedBackend,
      rowCount: rows.length,
      totalTimeMs: performance.now() - startTime,
      steps,
      activation,
      head: headId,
    };

    return {
      modelId: this.model.modelId,
      modelHash: this.model.modelHash || null,
      backend: usedBackend,
      head: headId,
      activation,
      rows: outputRows,
      totalTimeMs: this.stats.totalTimeMs,
    };
  }

  async infer(request = {}) {
    return this.scoreRows(request);
  }
}

export async function createDreamEnergyHeadPipeline(manifest, contexts = {}) {
  return createInitializedPipeline(DreamEnergyHeadPipeline, manifest, contexts);
}

for (const modelType of DREAM_ENERGY_HEAD_MODEL_TYPES) {
  registerPipeline(modelType, createDreamEnergyHeadPipeline);
}
