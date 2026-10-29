import { createTrainingConfig } from '../../config/training-defaults.js';
import { loadBackwardRegistry } from '../../config/backward-registry-loader.js';
import { getWeightDtype, isWeightBuffer } from '../../gpu/weight-buffer.js';
import { createTensor, tensorBytes } from '../../gpu/tensor.js';
import { runMatmul } from '../../gpu/kernels/matmul.js';
import { runRMSNorm } from '../../gpu/kernels/rmsnorm.js';
import { runResidualAdd } from '../../gpu/kernels/residual.js';
import { runScale } from '../../gpu/kernels/scale.js';
import { readBuffer, releaseBuffer, uploadData } from '../../memory/buffer-pool.js';
import { LoraAdapter } from './lora.js';
import { AdamOptimizer } from './optimizer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { trainStep } from './trainer.js';
import { OpType } from './autograd.js';
import { createUploadedTensor } from './tensor-factory.js';
import { exportLoRAAdapter } from './export.js';
import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { detectOverflow } from './loss-scaling.js';
import { createLoRATrainingRound, validateLoRATrainingRound, validateLoRAContributions, describeLoRATrainingSample } from './lora-training-round.js';

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

function validateTensorSnapshot(tensor, snapshot, label) {
  if (snapshot?.dtype !== tensor.dtype || JSON.stringify(snapshot?.shape) !== JSON.stringify(tensor.shape)) {
    throw new Error(`native_qwen_lora_checkpoint_tensor_mismatch: ${label}.`);
  }
  const bytes = decodeBytes(snapshot.bytes);
  const expectedBytes = tensorBytes(tensor.shape, tensor.dtype);
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`native_qwen_lora_checkpoint_tensor_size_mismatch: ${label}.`);
  }
  return bytes;
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
    this.operationActive = false;
    this.failed = false;
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

  async withOperation(operation) {
    if (this.disposed || this.failed) throw new Error('NativeQwenLoRATrainer is unavailable; initialize a new trainer.');
    if (this.operationActive) throw new Error('native_qwen_lora_operation_already_active.');
    this.operationActive = true;
    try { return await operation(); } finally { this.operationActive = false; }
  }

  trainStep(inputIds, targetIds, supervisedTokenCount) {
    return this.withOperation(() => this.runStep(inputIds, targetIds, supervisedTokenCount));
  }

  async runStep(inputIds, targetIds, supervisedTokenCount, contribution = null) {
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
          applyClip: contribution === null,
          applyOptimizer: contribution === null,
        }
      );
      try {
        if (contribution) {
          if (await detectOverflow(result.grads)) throw new Error('LoRA round: non-finite local gradient.');
          return {
            ...contribution,
            loss: result.objectiveMetrics.loss,
            gradients: {
              A: await serializeTensor(result.grads.get(this.adapter.A)),
              B: await serializeTensor(result.grads.get(this.adapter.B)),
            },
          };
        }
        return {
          loss: result.objectiveMetrics.loss,
          gradientNorm: result.clipMetrics?.gradient_norm_unclipped ?? null,
          optimizer: result.optimizerMetrics,
        };
      } finally {
        releaseTensorSet(result.grads.values(), new Set());
      }
    } finally {
      releaseBuffer(targets.buffer);
      if (this.activeCapture) {
        this.activeCapture.dispose();
        this.activeCapture = null;
      }
    }
  }

  exportAdapter(options) {
    return this.withOperation(() => exportLoRAAdapter({
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
      version: options.version,
      description: options.description,
      metadata: {
        ...options.metadata,
        baseManifestHash: computeCanonicalSha256(this.pipeline.manifest),
        trainingBackend: 'webgpu_native',
        trainingSurfaces: ['browser', 'node', 'bun'],
        targetLayer: this.layerIdx,
      },
    }));
  }

  createCheckpoint() {
    return this.withOperation(() => this.captureCheckpoint());
  }

  async captureCheckpoint() {
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
    if (this.operationActive) throw new Error('native_qwen_lora_operation_already_active.');
    if (this.failed) throw new Error('NativeQwenLoRATrainer is unavailable; initialize a new trainer.');
    this.applyCheckpoint(checkpoint);
  }

  applyCheckpoint(checkpoint) {
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
    const entries = [
      [this.adapter.A, checkpoint.adapter?.A, 'adapter.A'],
      [this.adapter.B, checkpoint.adapter?.B, 'adapter.B'],
    ];
    [this.adapter.A, this.adapter.B].forEach((param, index) => {
      const state = this.optimizer.getState(param);
      entries.push([state.m, checkpoint.optimizer[index]?.m, `optimizer.${index}.m`]);
      entries.push([state.v, checkpoint.optimizer[index]?.v, `optimizer.${index}.v`]);
    });
    const validated = entries.map(([tensor, snapshot, label]) => [tensor, validateTensorSnapshot(tensor, snapshot, label)]);
    for (const [tensor, bytes] of validated) uploadData(tensor.buffer, bytes);
    this.optimizer.stepCount = checkpoint.stepCount;
  }

  describeTrainingSample(participantId, sample) {
    return describeLoRATrainingSample(participantId, sample);
  }

  createTrainingRound({ jobId, roundId, assignments }) {
    return this.withOperation(async () => createLoRATrainingRound({
      jobId, roundId, assignments,
      baseManifestHash: computeCanonicalSha256(this.pipeline.manifest),
      trainingConfigHash: computeCanonicalSha256(this.trainingConfig.training),
      checkpoint: await this.captureCheckpoint(),
    }));
  }

  validateRoundIdentity(round) {
    validateLoRATrainingRound(round);
    if (round.baseManifestHash !== computeCanonicalSha256(this.pipeline.manifest)
      || round.trainingConfigHash !== computeCanonicalSha256(this.trainingConfig.training)) {
      throw new Error('LoRA round: base artifact or training configuration mismatch.');
    }
  }

  computeGradientContribution(round, participantId, sample) {
    return this.withOperation(async () => {
      this.validateRoundIdentity(round);
      const assignment = round.assignments.find((entry) => entry.participantId === participantId);
      if (!assignment || assignment.sampleHash !== computeCanonicalSha256(sample)
        || assignment.supervisedTokenCount !== sample.supervisedTokenCount) {
        throw new Error('LoRA round: sample does not match its assignment.');
      }
      this.applyCheckpoint(round.checkpoint);
      return this.runStep(sample.inputIds, sample.targetIds, sample.supervisedTokenCount, {
        schema: 'doppler.lora-gradient-contribution/v1',
        roundHash: round.roundHash,
        participantId,
        sampleHash: assignment.sampleHash,
        supervisedTokenCount: sample.supervisedTokenCount,
      });
    });
  }

  applyGradientContributions(round, contributions) {
    return this.withOperation(() => this.applyRound(round, contributions));
  }

  async applyRound(round, contributions) {
    this.validateRoundIdentity(round);
    const ordered = validateLoRAContributions(round, contributions);
    if (computeCanonicalSha256(await this.captureCheckpoint()) !== round.checkpointHash) {
      throw new Error('LoRA round: stale coordinator checkpoint; refusing a repeated or outdated update.');
    }
    const tokenCount = ordered.reduce((sum, entry) => sum + entry.supervisedTokenCount, 0);
    const parameters = [this.adapter.A, this.adapter.B];
    const owned = new Set();
    const gradients = new Map();
    let optimizerStarted = false;
    try {
      for (const [index, name] of ['A', 'B'].entries()) {
        const parameter = parameters[index];
        let sum = null;
        for (const entry of ordered) {
          const snapshot = entry.gradients?.[name];
          if (snapshot?.dtype !== 'f32' || JSON.stringify(snapshot.shape) !== JSON.stringify(parameter.shape)) {
            throw new Error(`LoRA round: invalid ${name} gradient layout.`);
          }
          const expectedBytes = tensorBytes(parameter.shape, 'f32');
          if (typeof snapshot.bytes !== 'string' || snapshot.bytes.length !== Math.ceil(expectedBytes / 3) * 4) {
            throw new Error('LoRA round: invalid gradient payload length.');
          }
          const bytes = decodeBytes(snapshot.bytes);
          if (bytes.byteLength !== expectedBytes) throw new Error('LoRA round: invalid gradient byte length.');
          const uploaded = createUploadedTensor(bytes, 'f32', parameter.shape, `lora_round_${name}`);
          owned.add(uploaded);
          const scaled = await runScale(uploaded, entry.supervisedTokenCount / tokenCount, { inplace: false });
          owned.add(scaled);
          sum = sum ? await runResidualAdd(sum, scaled, parameter.shape.reduce((a, b) => a * b, 1)) : scaled;
          owned.add(sum);
        }
        gradients.set(parameter, sum);
      }
      if (await detectOverflow(gradients)) throw new Error('LoRA round: non-finite aggregate gradient.');
      // Clip the global token-weighted mean exactly once, then perform one AdamW update.
      const clipped = await clipGradients(gradients, this.trainingConfig);
      for (const tensor of clipped.clippedGrads.values()) owned.add(tensor);
      optimizerStarted = true;
      const optimizer = await this.optimizer.step(parameters, clipped.clippedGrads, this.trainingConfig);
      return {
        roundHash: round.roundHash,
        contributors: ordered.map((entry) => entry.participantId),
        supervisedTokenCount: tokenCount,
        loss: ordered.reduce((sum, entry) => sum + entry.loss * (entry.supervisedTokenCount / tokenCount), 0),
        gradientNorm: clipped.gradient_norm_unclipped,
        optimizer,
        stepCount: this.optimizer.stepCount,
      };
    } catch (error) {
      // A failed GPU update may be partial. Never continue or silently retry it.
      if (optimizerStarted) this.failed = true;
      throw error;
    } finally {
      releaseTensorSet(owned, new Set());
    }
  }

  dispose() {
    if (this.disposed) return;
    if (this.operationActive) throw new Error('native_qwen_lora_operation_already_active.');
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
