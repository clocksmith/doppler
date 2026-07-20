import { createTrainingConfig } from '../../config/training-defaults.js';
import { loadBackwardRegistry } from '../../config/backward-registry-loader.js';
import { getWeightDtype, isWeightBuffer } from '../../gpu/weight-buffer.js';
import { createTensor, tensorBytes } from '../../gpu/tensor.js';
import { runMatmul, runRMSNorm, runResidualAdd } from '../../gpu/kernels/index.js';
import { readBuffer, releaseBuffer, uploadData } from '../../memory/buffer-pool.js';
import { LoraAdapter } from './lora.js';
import { AdamOptimizer } from './optimizer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { trainStep } from './trainer.js';
import { OpType } from './autograd.js';
import { createUploadedTensor } from './tensor-factory.js';
import { exportLoRAAdapter } from './export.js';

function asWeightTensor(weight, shape, label) {
  if (!weight) throw new Error(`native_qwen_lora_missing_weight: ${label}.`);
  if (isWeightBuffer(weight)) {
    return { ...weight, shape: [...shape], label };
  }
  const dtype = getWeightDtype(weight) || 'f32';
  return createTensor(weight, dtype, [...shape], label);
}

function releaseTensorSet(tensors, preserved) {
  const released = new Set();
  for (const tensor of tensors) {
    const buffer = tensor?.buffer;
    if (!buffer || preserved.has(buffer) || released.has(buffer)) continue;
    released.add(buffer);
    releaseBuffer(buffer);
  }
}

function encodeBytes(bytes) {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (typeof btoa === 'function') {
    let binary = '';
    for (let index = 0; index < view.length; index += 1) binary += String.fromCharCode(view[index]);
    return btoa(binary);
  }
  if (typeof Buffer !== 'undefined') return Buffer.from(view).toString('base64');
  throw new Error('native_qwen_lora_checkpoint_base64_unavailable.');
}

function decodeBytes(value) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error('native_qwen_lora_checkpoint_bytes_invalid.');
  }
  if (typeof atob === 'function') {
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return bytes;
  }
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(value, 'base64'));
  throw new Error('native_qwen_lora_checkpoint_base64_unavailable.');
}

async function serializeTensor(tensor) {
  const byteLength = tensorBytes(tensor.shape, tensor.dtype);
  return {
    dtype: tensor.dtype,
    shape: [...tensor.shape],
    bytes: encodeBytes(await readBuffer(tensor.buffer, byteLength)),
  };
}

function restoreTensor(tensor, snapshot, label) {
  if (snapshot?.dtype !== tensor.dtype || JSON.stringify(snapshot?.shape) !== JSON.stringify(tensor.shape)) {
    throw new Error(`native_qwen_lora_checkpoint_tensor_mismatch: ${label}.`);
  }
  const bytes = decodeBytes(snapshot.bytes);
  const expectedBytes = tensorBytes(tensor.shape, tensor.dtype);
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`native_qwen_lora_checkpoint_tensor_size_mismatch: ${label}.`);
  }
  uploadData(tensor.buffer, bytes);
}

function createCausalLmObjective(owner) {
  return {
    name: 'native_qwen_completion_masked_sft',
    async forward({ batch, tape }) {
      return owner.forward(batch.inputIds, tape);
    },
    async computeLoss({ batch, config, tape, forwardState }) {
      return {
        loss: await crossEntropyLoss(forwardState.logits, batch.targets, config, tape),
        components: {
          supervised_token_count: batch.supervisedTokenCount,
          ignored_target_count: batch.targets.shape[0] - batch.supervisedTokenCount,
        },
      };
    },
    backwardTargets({ batch, loss, lossScale }) {
      const values = new Float32Array(loss.shape[0]);
      values.fill(lossScale / batch.supervisedTokenCount);
      return createUploadedTensor(values, 'f32', loss.shape, 'native_qwen_lora_loss_grad');
    },
    async metrics({ batch, loss }) {
      const bytes = await readBuffer(loss.buffer, loss.shape[0] * Float32Array.BYTES_PER_ELEMENT);
      const values = new Float32Array(bytes);
      let sum = 0;
      for (const value of values) sum += value;
      return {
        loss: sum / batch.supervisedTokenCount,
        supervised_token_count: batch.supervisedTokenCount,
      };
    },
    cleanup({ tape }) {
      owner.cleanupStep(tape);
    },
  };
}

export class NativeQwenLoRATrainer {
  constructor(options) {
    const pipeline = options?.pipeline;
    const config = pipeline?.modelConfig;
    if (!pipeline || !config || !(pipeline.weights instanceof Map)) {
      throw new Error('NativeQwenLoRATrainer requires a loaded Doppler text pipeline.');
    }
    const layerIdx = Number(options.layerIdx);
    if (!Number.isInteger(layerIdx) || layerIdx !== config.numLayers - 1) {
      throw new Error(`native_lora_target_not_supported: layerIdx must be ${config.numLayers - 1}.`);
    }
    if (options.module !== 'down_proj') {
      throw new Error('native_lora_target_not_supported: module must be down_proj.');
    }
    if (config.layerTypes?.[layerIdx] !== 'full_attention' || config.postFeedforwardNorm === true) {
      throw new Error('native_lora_target_not_supported: target requires a final full-attention layer without post-FFN normalization.');
    }
    const lmHeadSource = pipeline.weights.get('lm_head')
      || (config.useTiedEmbeddings === true ? pipeline.weights.get('embed') : null);
    const lmHead = asWeightTensor(lmHeadSource, [config.vocabSize, config.hiddenSize], 'native_qwen_lm_head');
    if (lmHead.dtype !== 'q4k' || lmHead.layout !== 'row') {
      throw new Error('native_lora_target_not_supported: Qwen LM head must be row-wise Q4K [vocab,hidden].');
    }

    this.pipeline = pipeline;
    this.config = config;
    this.layerIdx = layerIdx;
    this.module = 'down_proj';
    this.baseModelId = String(options.baseModelId || pipeline.manifest?.modelId || '');
    this.lmHead = lmHead;
    this.finalNorm = asWeightTensor(
      pipeline.weights.get('final_norm'),
      [config.hiddenSize],
      'native_qwen_final_norm'
    );
    this.trainingConfig = createTrainingConfig({
      training: {
        enabled: true,
        optimizer: options.optimizer,
        gradient: options.gradient,
        precision: options.precision,
        lossScaling: { enabled: false },
        distill: { freeze: { base: true, lora: false } },
      },
    });
    this.adapter = new LoraAdapter({
      inDim: config.intermediateSize,
      outDim: config.hiddenSize,
      rank: options.rank,
      alpha: options.alpha,
      dtype: this.trainingConfig.training.precision.loraParams,
    });
    this.optimizer = new AdamOptimizer(this.trainingConfig);
    this.registry = loadBackwardRegistry();
    this.activeCapture = null;
    this.disposed = false;
  }

  paramGroups() {
    return {
      base: [this.finalNorm, this.lmHead],
      lora: [this.adapter.A, this.adapter.B],
    };
  }

  async forward(inputIds, tape) {
    if (this.activeCapture) throw new Error('native_qwen_lora_step_already_active.');
    const capture = await this.pipeline.prefillForLoRATraining(inputIds, {
      layerIdx: this.layerIdx,
      module: this.module,
    });
    this.activeCapture = capture;
    const delta = await this.adapter.forward(capture.activation, tape);
    const adaptedHidden = await tape.record(
      OpType.RESIDUAL_ADD,
      (base, update) => runResidualAdd(base, update, inputIds.length * this.config.hiddenSize),
      [capture.baseHidden, delta],
      { size: inputIds.length * this.config.hiddenSize, stopGradInputs: [0] }
    );
    const finalHidden = await tape.record(
      OpType.RMSNORM,
      (hidden, weight) => runRMSNorm(hidden, weight, this.config.rmsNormEps, {
        batchSize: inputIds.length,
        hiddenSize: this.config.hiddenSize,
        rmsNormWeightOffset: this.config.rmsNormWeightOffset === true,
      }),
      [adaptedHidden, this.finalNorm],
      {
        numTokens: inputIds.length,
        hiddenSize: this.config.hiddenSize,
        eps: this.config.rmsNormEps,
        rmsNormWeightOffset: this.config.rmsNormWeightOffset === true,
        stopGradInputs: [1],
      }
    );
    const logits = await tape.record(
      OpType.MATMUL,
      (hidden, weight) => runMatmul(
        hidden,
        weight,
        inputIds.length,
        this.config.vocabSize,
        this.config.hiddenSize,
        {
          transposeB: true,
          outputDtype: 'f32',
          role: 'lm_head_prefill',
          phaseOverride: 'prefill',
          kernelPath: this.pipeline.resolvedKernelPath,
          executionPolicies: this.pipeline.executionV1State?.policies ?? null,
        }
      ),
      [finalHidden, this.lmHead],
      {
        M: inputIds.length,
        N: this.config.vocabSize,
        K: this.config.hiddenSize,
        transposeB: true,
        stopGradInputs: [1],
      }
    );
    return { logits };
  }

  cleanupStep(tape) {
    const capture = this.activeCapture;
    const preserved = new Set([
      this.adapter.A.buffer,
      this.adapter.B.buffer,
      this.finalNorm.buffer,
      this.lmHead.buffer,
      capture?.activation?.buffer,
      capture?.baseHidden?.buffer,
    ]);
    releaseTensorSet(tape.records.map((record) => record.output), preserved);
    capture?.dispose();
    this.activeCapture = null;
  }

  async trainStep(inputIds, targetIds, supervisedTokenCount) {
    if (this.disposed) throw new Error('NativeQwenLoRATrainer is disposed.');
    if (inputIds.length !== targetIds.length) {
      throw new Error('native_qwen_lora_target_length_mismatch.');
    }
    if (!Number.isInteger(supervisedTokenCount) || supervisedTokenCount < 1) {
      throw new Error('native_qwen_lora_supervised_token_count_invalid.');
    }
    const actualSupervisedTokenCount = Array.from(targetIds)
      .filter((target) => Number.isInteger(target) && target >= 0 && target < this.config.vocabSize)
      .length;
    if (actualSupervisedTokenCount !== supervisedTokenCount) {
      throw new Error(
        `native_qwen_lora_supervised_token_count_mismatch: expected ${actualSupervisedTokenCount}, got ${supervisedTokenCount}.`
      );
    }
    const targets = createUploadedTensor(
      new Uint32Array(targetIds),
      'f32',
      [targetIds.length],
      'native_qwen_lora_targets'
    );
    try {
      const result = await trainStep(
        this,
        { inputIds: Array.from(inputIds), targets, supervisedTokenCount },
        this.trainingConfig,
        {
          registry: this.registry,
          crossEntropyLoss,
          clipGradients,
          optimizer: this.optimizer,
          trainingObjective: createCausalLmObjective(this),
        }
      );
      const gradientNorm = result.clipMetrics?.gradient_norm_unclipped ?? null;
      releaseTensorSet(result.grads.values(), new Set());
      return {
        loss: result.objectiveMetrics.loss,
        gradientNorm,
        optimizer: result.optimizerMetrics,
      };
    } finally {
      releaseBuffer(targets.buffer);
      if (this.activeCapture) {
        this.activeCapture.dispose();
        this.activeCapture = null;
      }
    }
  }

  exportAdapter(options) {
    return exportLoRAAdapter({
      id: options.id,
      name: options.name,
      baseModel: this.baseModelId,
      rank: this.adapter.rank,
      alpha: this.adapter.alpha,
      targetModules: ['down_proj'],
      tensors: [
        { name: `layers.${this.layerIdx}.down_proj.lora_a`, tensor: this.adapter.A },
        { name: `layers.${this.layerIdx}.down_proj.lora_b`, tensor: this.adapter.B },
      ],
      weightsFormat: 'safetensors',
      weightsPath: options.weightsPath,
      metadata: {
        trainingBackend: 'webgpu_native',
        trainingSurfaces: ['browser', 'node', 'bun'],
        targetLayer: this.layerIdx,
      },
    });
  }

  async createCheckpoint() {
    if (this.disposed) throw new Error('NativeQwenLoRATrainer is disposed.');
    const params = [this.adapter.A, this.adapter.B];
    const optimizer = [];
    for (const param of params) {
      const state = this.optimizer.getState(param);
      optimizer.push({
        m: await serializeTensor(state.m),
        v: await serializeTensor(state.v),
      });
    }
    return {
      schemaVersion: 1,
      backend: 'webgpu_native',
      baseModelId: this.baseModelId,
      target: { layerIdx: this.layerIdx, module: this.module },
      rank: this.adapter.rank,
      alpha: this.adapter.alpha,
      stepCount: this.optimizer.stepCount,
      adapter: {
        A: await serializeTensor(this.adapter.A),
        B: await serializeTensor(this.adapter.B),
      },
      optimizer,
    };
  }

  restoreCheckpoint(checkpoint) {
    if (this.disposed) throw new Error('NativeQwenLoRATrainer is disposed.');
    if (
      checkpoint?.schemaVersion !== 1
      || checkpoint.backend !== 'webgpu_native'
      || checkpoint.baseModelId !== this.baseModelId
      || checkpoint.target?.layerIdx !== this.layerIdx
      || checkpoint.target?.module !== this.module
      || checkpoint.rank !== this.adapter.rank
      || checkpoint.alpha !== this.adapter.alpha
      || !Number.isInteger(checkpoint.stepCount)
      || checkpoint.stepCount < 0
      || !Array.isArray(checkpoint.optimizer)
      || checkpoint.optimizer.length !== 2
    ) {
      throw new Error('native_qwen_lora_checkpoint_identity_mismatch.');
    }
    const params = [this.adapter.A, this.adapter.B];
    restoreTensor(this.adapter.A, checkpoint.adapter?.A, 'adapter.A');
    restoreTensor(this.adapter.B, checkpoint.adapter?.B, 'adapter.B');
    params.forEach((param, index) => {
      const state = this.optimizer.getState(param);
      restoreTensor(state.m, checkpoint.optimizer[index]?.m, `optimizer.${index}.m`);
      restoreTensor(state.v, checkpoint.optimizer[index]?.v, `optimizer.${index}.v`);
    });
    this.optimizer.stepCount = checkpoint.stepCount;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.activeCapture) this.activeCapture.dispose();
    for (const state of this.optimizer.state.values()) {
      releaseBuffer(state.m.buffer);
      releaseBuffer(state.v.buffer);
    }
    this.optimizer.state.clear();
    this.adapter.dispose();
  }
}

export function createNativeQwenLoRATrainer(options) {
  return new NativeQwenLoRATrainer(options);
}

export async function loadNativeQwenTrainingPipeline(modelUrl, options = {}) {
  if (typeof modelUrl !== 'string' || modelUrl.trim().length === 0) {
    throw new Error('loadNativeQwenTrainingPipeline requires an RDRR model URL.');
  }
  const { initializeInference } = await import('../../inference/test-harness.js');
  const loaded = await initializeInference(modelUrl.replace(/\/$/, ''), options);
  if (loaded.manifest?.modelId !== 'qwen-3-5-0-8b-q4k-ehaf16') {
    await loaded.pipeline.unload();
    throw new Error(
      `native_qwen_lora_model_not_supported: expected qwen-3-5-0-8b-q4k-ehaf16, got ${String(loaded.manifest?.modelId)}.`
    );
  }
  return loaded;
}

export async function trainNativeQwenSftLoRA(options) {
  if (!Array.isArray(options?.samples) || options.samples.length === 0) {
    throw new Error('trainNativeQwenSftLoRA requires tokenized samples.');
  }
  const trainer = createNativeQwenLoRATrainer(options);
  const metrics = [];
  try {
    const maxSteps = options.maxSteps === undefined ? options.samples.length : Number(options.maxSteps);
    if (!Number.isInteger(maxSteps) || maxSteps < 1) {
      throw new Error('trainNativeQwenSftLoRA maxSteps must be a positive integer.');
    }
    for (let step = 0; step < maxSteps; step += 1) {
      const sample = options.samples[step % options.samples.length];
      metrics.push(await trainer.trainStep(
        sample.inputIds,
        sample.targetIds,
        sample.supervisedTokenCount
      ));
    }
    const adapter = options.export
      ? await trainer.exportAdapter(options.export)
      : null;
    return {
      backend: 'webgpu_native',
      surfaces: ['browser', 'node', 'bun'],
      baseModelId: trainer.baseModelId,
      target: { layerIdx: trainer.layerIdx, module: trainer.module },
      metrics,
      adapter,
    };
  } finally {
    trainer.dispose();
  }
}
