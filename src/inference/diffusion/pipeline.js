import { getDevice, setDevice, getKernelCapabilities } from '../../gpu/device.js';
import { log, trace, applyDebugConfig, setGPUDevice } from '../../debug/index.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import { registerPipeline } from '../pipeline/registry.js';
import { initializeDiffusion } from './init.js';
import { loadDiffusionTokenizers, encodePrompt } from './text-encoder.js';
import {
  runTextEncodersForPrompt,
  buildTimeTextEmbedding,
  buildTimestepEmbedding,
  combineTimeTextEmbeddings,
  projectContext,
  logQuickGeluWarning,
} from './text-encoder-gpu.js';
import { buildScheduler } from './scheduler.js';
import { runUnetStep } from './unet.js';
import { decodeLatents } from './vae.js';
import { initializeDiffusionGpuScaffold, runDiffusionGpuScaffold, logDiffusionGpuScaffold } from './gpu-ops.js';
import { createDiffusionWeightLoader } from './weights.js';
import { runSD3Transformer } from './sd3-transformer.js';
import { createSD3WeightResolver } from './sd3-weights.js';
import { createTensor, dtypeBytes } from '../../gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { CommandRecorder } from '../../gpu/command-recorder.js';
import { castF32ToF16 } from '../../gpu/kernels/cast.js';
import { runResidualAdd, runScale, recordResidualAdd, recordScale } from '../../gpu/kernels/index.js';
import { f16ToF32 } from '../../loader/dtype-utils.js';

function createRng(seed) {
  let state = seed >>> 0;
  if (!state) state = 0x6d2b79f5;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateLatents(width, height, channels, latentScale, seed) {
  const latentWidth = Math.max(1, Math.floor(width / latentScale));
  const latentHeight = Math.max(1, Math.floor(height / latentScale));
  const size = latentWidth * latentHeight * channels;
  const latents = new Float32Array(size);
  const rand = createRng(seed ?? Math.floor(Math.random() * 1e9));
  for (let i = 0; i < size; i++) {
    const u = rand();
    const v = rand();
    const z = Math.sqrt(-2.0 * Math.log(Math.max(u, 1e-6))) * Math.cos(2.0 * Math.PI * v);
    latents[i] = z;
  }
  return { latents, latentWidth, latentHeight };
}

function extractTokenSet(tokensByEncoder, key) {
  const output = {};
  for (const [name, entry] of Object.entries(tokensByEncoder || {})) {
    const tokens = entry?.[key];
    output[name] = Array.isArray(tokens) ? tokens : [];
  }
  return output;
}

function getTensorSize(shape) {
  if (!Array.isArray(shape)) return 0;
  return shape.reduce((acc, value) => acc * value, 1);
}

function sumProfileTimings(timings) {
  if (!timings) return null;
  return Object.values(timings).reduce((sum, value) => sum + value, 0);
}

function createRecorderReleaser(recorder) {
  if (!recorder) {
    return (buffer) => {
      if (!buffer) return;
      releaseBuffer(buffer);
    };
  }
  return (buffer) => {
    if (!buffer) return;
    recorder.trackTemporaryBuffer(buffer);
  };
}

async function createLatentTensor(latents, shape, runtime) {
  const device = getDevice();
  if (!device) {
    throw new Error('Diffusion GPU path requires a WebGPU device.');
  }
  const buffer = acquireBuffer(latents.byteLength, undefined, 'diffusion_latents');
  device.queue.writeBuffer(buffer, 0, latents);
  let tensor = createTensor(buffer, 'f32', shape, 'diffusion_latents_f32');

  const wantsF16 = runtime?.latent?.dtype === 'f16';
  const caps = getKernelCapabilities();
  if (wantsF16 && caps.hasF16) {
    const casted = await castF32ToF16(tensor);
    releaseBuffer(tensor.buffer);
    tensor = casted;
  } else if (wantsF16 && !caps.hasF16) {
    log.warn('Diffusion', 'Requested f16 latents but device lacks f16 support. Using f32.');
  }

  return tensor;
}

async function readTensorToFloat32(tensor) {
  const size = getTensorSize(tensor.shape);
  const byteLength = size * dtypeBytes(tensor.dtype);
  const data = await readBuffer(tensor.buffer, byteLength);

  if (tensor.dtype === 'f16') {
    const u16 = new Uint16Array(data);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      out[i] = f16ToF32(u16[i]);
    }
    return out;
  }

  return new Float32Array(data);
}

async function applyGuidance(uncond, cond, guidanceScale, size, options = {}) {
  if (!uncond || !Number.isFinite(guidanceScale) || guidanceScale <= 1) {
    return cond;
  }

  const recorder = options.recorder ?? null;
  const release = options.release ?? createRecorderReleaser(recorder);
  const scale = recorder
    ? (input, scalar, opts) => recordScale(recorder, input, scalar, opts)
    : runScale;
  const residualAdd = recorder
    ? (left, right, count, opts) => recordResidualAdd(recorder, left, right, count, opts)
    : runResidualAdd;

  const negUncond = await scale(uncond, -1, { count: size });
  const diff = await residualAdd(cond, negUncond, size, { useVec4: true });
  release(negUncond.buffer);

  const diffTensor = createTensor(diff.buffer, diff.dtype, [...cond.shape], 'sd3_guidance_diff');
  const scaled = await scale(diffTensor, guidanceScale, { count: size });
  release(diffTensor.buffer);

  const guided = await residualAdd(uncond, scaled, size, { useVec4: true });
  release(scaled.buffer);

  return createTensor(guided.buffer, guided.dtype, [...cond.shape], 'sd3_guided');
}

export class DiffusionPipeline {
  
  runtimeConfig = null;
  
  manifest = null;
  
  diffusionState = null;
  
  tokenizers = null;
  
  stats = {};
  
  baseUrl = null;
  
  _onProgress = null;
  gpuScaffold = null;
  weightLoader = null;
  vaeWeights = null;
  textEncoderWeights = null;
  transformerWeights = null;

  async initialize(contexts = {}) {
    if (contexts.runtimeConfig) {
      this.runtimeConfig = setRuntimeConfig(contexts.runtimeConfig);
    } else {
      this.runtimeConfig = getRuntimeConfig();
    }
    const sharedDebug = this.runtimeConfig.shared?.debug;
    if (sharedDebug) {
      applyDebugConfig(sharedDebug);
    }

    if (contexts.gpu?.device) {
      setDevice(contexts.gpu.device);
      setGPUDevice(contexts.gpu.device);
    } else {
      const device = getDevice();
      if (device) setGPUDevice(device);
    }

    if (contexts.baseUrl) this.baseUrl = contexts.baseUrl;
    if (contexts.onProgress) this._onProgress = contexts.onProgress;
  }

  async loadModel(manifest) {
    if (!manifest || manifest.modelType !== 'diffusion') {
      throw new Error('Diffusion pipeline requires a diffusion model manifest.');
    }
    this.manifest = manifest;
    this.diffusionState = initializeDiffusion(manifest, this.runtimeConfig);
    this.tokenizers = await loadDiffusionTokenizers(this.diffusionState.modelConfig, {
      baseUrl: this.baseUrl,
    });
    log.info('Diffusion', `Loaded diffusion model "${manifest.modelId}" with ${Object.keys(this.tokenizers || {}).length} tokenizers`);
    this.weightLoader = await createDiffusionWeightLoader(manifest, {
      baseUrl: this.baseUrl,
      runtimeConfig: this.runtimeConfig,
    });
    const pipelineMode = this.diffusionState.runtime?.backend?.pipeline;
    if (pipelineMode === 'gpu_scaffold') {
      this.gpuScaffold = initializeDiffusionGpuScaffold(this.diffusionState.runtime);
      logDiffusionGpuScaffold(this.gpuScaffold);
    } else if (pipelineMode === 'gpu') {
      log.info('Diffusion', 'GPU diffusion pipeline enabled.');
    } else {
      log.warn('Diffusion', 'Diffusion kernels are not implemented yet; using CPU placeholder pipeline.');
    }
  }

  getStats() {
    return this.stats;
  }

  getMemoryStats() {
    return {
      used: 0,
      kvCache: null,
    };
  }

  async unload() {
    this.vaeWeights?.release?.();
    this.textEncoderWeights?.text_encoder?.release?.();
    this.textEncoderWeights?.text_encoder_2?.release?.();
    this.textEncoderWeights?.text_encoder_3?.release?.();
    this.transformerWeights?.release?.();
    this.tokenizers = null;
    this.manifest = null;
    this.diffusionState = null;
    this.gpuScaffold = null;
    this.weightLoader = null;
    this.vaeWeights = null;
    this.textEncoderWeights = null;
    this.transformerWeights = null;
  }

  async ensureVaeWeights() {
    if (this.vaeWeights) return;
    if (!this.weightLoader) {
      if (!this.manifest) throw new Error('Diffusion weight loader not initialized.');
      this.weightLoader = await createDiffusionWeightLoader(this.manifest, {
        baseUrl: this.baseUrl,
        runtimeConfig: this.runtimeConfig,
      });
    }
    this.vaeWeights = await this.weightLoader.loadComponentWeights('vae', {
      filter: (name) => (
        name.startsWith('vae.decoder.') ||
        name.startsWith('vae.quant_conv.') ||
        name.startsWith('vae.post_quant_conv.')
      ),
    });
  }

  async ensureTextEncoderWeights() {
    if (this.textEncoderWeights) return this.textEncoderWeights;
    if (!this.weightLoader) {
      if (!this.manifest) throw new Error('Diffusion weight loader not initialized.');
      this.weightLoader = await createDiffusionWeightLoader(this.manifest, {
        baseUrl: this.baseUrl,
        runtimeConfig: this.runtimeConfig,
      });
    }

    const text_encoder = await this.weightLoader.loadComponentWeights('text_encoder');
    const text_encoder_2 = await this.weightLoader.loadComponentWeights('text_encoder_2');
    const text_encoder_3 = await this.weightLoader.loadComponentWeights('text_encoder_3');

    this.textEncoderWeights = {
      text_encoder,
      text_encoder_2,
      text_encoder_3,
    };

    return this.textEncoderWeights;
  }

  async ensureTransformerWeights() {
    if (this.transformerWeights) return this.transformerWeights;
    if (!this.weightLoader) {
      if (!this.manifest) throw new Error('Diffusion weight loader not initialized.');
      this.weightLoader = await createDiffusionWeightLoader(this.manifest, {
        baseUrl: this.baseUrl,
        runtimeConfig: this.runtimeConfig,
      });
    }
    this.transformerWeights = await this.weightLoader.loadComponentWeights('transformer');
    return this.transformerWeights;
  }

  releaseTextEncoderWeights() {
    if (!this.textEncoderWeights) return;
    this.textEncoderWeights.text_encoder?.release?.();
    this.textEncoderWeights.text_encoder_2?.release?.();
    this.textEncoderWeights.text_encoder_3?.release?.();
    this.textEncoderWeights = null;
  }

  releaseTransformerWeights() {
    if (!this.transformerWeights) return;
    this.transformerWeights.release?.();
    this.transformerWeights = null;
  }

  async generate(request = {}) {
    if (!this.diffusionState) {
      throw new Error('Diffusion pipeline not initialized.');
    }
    const pipelineMode = this.diffusionState.runtime?.backend?.pipeline;
    if (pipelineMode === 'gpu') {
      return this.generateGPU(request);
    }
    return this.generateCPU(request);
  }

  async generateCPU(request = {}) {
    const start = performance.now();
    const runtime = this.diffusionState.runtime;
    const clipMaxLength = runtime.textEncoder?.maxLength;
    if (!Number.isFinite(clipMaxLength) || clipMaxLength <= 0) {
      throw new Error('Diffusion runtime requires runtime.textEncoder.maxLength.');
    }
    const t5MaxLength = runtime.textEncoder?.t5MaxLength ?? clipMaxLength;
    if (!Number.isFinite(t5MaxLength) || t5MaxLength <= 0) {
      throw new Error('Diffusion runtime requires runtime.textEncoder.t5MaxLength (or runtime.textEncoder.maxLength).');
    }

    const defaultWidth = runtime.latent.width;
    const defaultHeight = runtime.latent.height;
    const width = Math.floor(Number.isFinite(request.width) && request.width > 0 ? request.width : defaultWidth);
    const height = Math.floor(Number.isFinite(request.height) && request.height > 0 ? request.height : defaultHeight);
    const steps = Math.floor(Number.isFinite(request.steps) && request.steps > 0 ? request.steps : runtime.scheduler.numSteps);
    const guidanceScale = Number.isFinite(request.guidanceScale) && request.guidanceScale > 0
      ? request.guidanceScale
      : runtime.scheduler.guidanceScale;
    const seed = Number.isFinite(request.seed) ? Math.floor(request.seed) : Math.floor(Math.random() * 1e9);

    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
      throw new Error(`Invalid diffusion dimensions: ${width}x${height}`);
    }
    if (!Number.isFinite(steps) || steps <= 0) {
      throw new Error(`Invalid diffusion steps: ${steps}`);
    }

    const promptStart = performance.now();
    const encoded = encodePrompt(
      { prompt: request.prompt ?? '', negativePrompt: request.negativePrompt ?? '' },
      this.tokenizers || {},
      {
        maxLengthByTokenizer: {
          text_encoder: clipMaxLength,
          text_encoder_2: clipMaxLength,
          text_encoder_3: t5MaxLength,
        },
      }
    );
    const promptEnd = performance.now();

    const scheduler = buildScheduler(runtime.scheduler, steps);
    const latentScale = this.diffusionState.latentScale;
    const latentChannels = this.diffusionState.latentChannels;
    const { latents, latentWidth, latentHeight } = generateLatents(width, height, latentChannels, latentScale, seed);

    this._onProgress?.({
      stage: 'diffusion',
      message: `Denoising ${scheduler.steps} steps...`,
      progress: 0,
    });

    const decodeStart = performance.now();
    for (let i = 0; i < scheduler.steps; i++) {
      if (this.gpuScaffold) {
        await runDiffusionGpuScaffold(this.gpuScaffold, { stepIndex: i });
      }
      runUnetStep(latents, scheduler, i, guidanceScale);
      if (i % 5 === 0 || i === scheduler.steps - 1) {
        this._onProgress?.({
          stage: 'diffusion',
          message: `Denoising ${i + 1}/${scheduler.steps}`,
          progress: (i + 1) / scheduler.steps,
        });
      }
    }
    const decodeEnd = performance.now();

    const vaeStart = performance.now();
    const pixels = await decodeLatents(latents, {
      width,
      height,
      latentWidth,
      latentHeight,
      latentChannels,
      latentScale,
      weights: null,
      modelConfig: this.diffusionState.modelConfig,
      runtime: this.diffusionState.runtime,
    });
    const vaeEnd = performance.now();

    const end = performance.now();
    const cpuPrefillMs = promptEnd - promptStart;
    const cpuDenoiseMs = decodeEnd - decodeStart;
    const cpuVaeMs = vaeEnd - vaeStart;

    this.stats = {
      totalTimeMs: end - start,
      prefillTimeMs: cpuPrefillMs,
      prefillTokens: encoded.totalTokens,
      decodeTimeMs: cpuDenoiseMs,
      decodeTokens: scheduler.steps,
      vaeTimeMs: cpuVaeMs,
      gpu: { available: false },
    };

    log.info('Diffusion', `Prompt encode: ${(promptEnd - promptStart).toFixed(0)}ms (${encoded.totalTokens} tokens)`);
    log.info('Diffusion', `Denoise: ${(decodeEnd - decodeStart).toFixed(0)}ms (${scheduler.steps} steps)`);
    log.info('Diffusion', `VAE decode: ${(vaeEnd - vaeStart).toFixed(0)}ms (${width}x${height})`);
    log.info('Diffusion', `Total: ${(end - start).toFixed(0)}ms`);
    trace.perf('Diffusion summary', {
      prefillMs: cpuPrefillMs,
      prefillTokens: encoded.totalTokens,
      denoiseMs: cpuDenoiseMs,
      steps: scheduler.steps,
      vaeMs: cpuVaeMs,
      totalMs: end - start,
      gpuPrefillMs: null,
      gpuDenoiseMs: null,
      gpuVaeMs: null,
      gpuTotalMs: null,
      width,
      height,
    });

    return { width, height, pixels };
  }

  async generateGPU(request = {}) {
    const start = performance.now();
    const runtime = this.diffusionState.runtime;
    const clipMaxLength = runtime.textEncoder?.maxLength;
    if (!Number.isFinite(clipMaxLength) || clipMaxLength <= 0) {
      throw new Error('Diffusion runtime requires runtime.textEncoder.maxLength.');
    }
    const t5MaxLength = runtime.textEncoder?.t5MaxLength ?? clipMaxLength;
    if (!Number.isFinite(t5MaxLength) || t5MaxLength <= 0) {
      throw new Error('Diffusion runtime requires runtime.textEncoder.t5MaxLength (or runtime.textEncoder.maxLength).');
    }

    const defaultWidth = runtime.latent.width;
    const defaultHeight = runtime.latent.height;
    const width = Math.floor(Number.isFinite(request.width) && request.width > 0 ? request.width : defaultWidth);
    const height = Math.floor(Number.isFinite(request.height) && request.height > 0 ? request.height : defaultHeight);
    const steps = Math.floor(Number.isFinite(request.steps) && request.steps > 0 ? request.steps : runtime.scheduler.numSteps);
    const guidanceScale = Number.isFinite(request.guidanceScale) && request.guidanceScale > 0
      ? request.guidanceScale
      : runtime.scheduler.guidanceScale;
    const seed = Number.isFinite(request.seed) ? Math.floor(request.seed) : Math.floor(Math.random() * 1e9);
    const profilerEnabled = this.runtimeConfig?.shared?.debug?.profiler?.enabled === true;
    const canProfileGpu = profilerEnabled && getKernelCapabilities().hasTimestampQuery;
    let gpuPrefillMs = canProfileGpu ? 0 : null;
    let gpuDenoiseMs = canProfileGpu ? 0 : null;
    let gpuVaeMs = canProfileGpu ? 0 : null;

    if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
      throw new Error(`Invalid diffusion dimensions: ${width}x${height}`);
    }
    if (!Number.isFinite(steps) || steps <= 0) {
      throw new Error(`Invalid diffusion steps: ${steps}`);
    }

    const modelConfig = this.diffusionState.modelConfig;
    if (!modelConfig?.components?.transformer) {
      throw new Error('Diffusion GPU pipeline requires transformer component config.');
    }
    if (!modelConfig?.components?.text_encoder || !modelConfig?.components?.text_encoder_2 || !modelConfig?.components?.text_encoder_3) {
      throw new Error('Diffusion GPU pipeline requires text encoder components (text_encoder, text_encoder_2, text_encoder_3).');
    }
    if (!this.tokenizers?.text_encoder || !this.tokenizers?.text_encoder_2 || !this.tokenizers?.text_encoder_3) {
      throw new Error('Diffusion GPU pipeline requires tokenizers for text_encoder, text_encoder_2, and text_encoder_3.');
    }
    logQuickGeluWarning(modelConfig?.components?.text_encoder?.config || {});

    const promptStart = performance.now();
    const encoded = encodePrompt(
      { prompt: request.prompt ?? '', negativePrompt: request.negativePrompt ?? '' },
      this.tokenizers || {},
      {
        maxLengthByTokenizer: {
          text_encoder: clipMaxLength,
          text_encoder_2: clipMaxLength,
          text_encoder_3: t5MaxLength,
        },
      }
    );

    const promptTokens = extractTokenSet(encoded.tokens, 'prompt');
    const negativeTokens = extractTokenSet(encoded.tokens, 'negative');
    const shouldUseUncond = guidanceScale > 1.0;

    const textWeights = await this.ensureTextEncoderWeights();
    const promptCondition = await runTextEncodersForPrompt(promptTokens, textWeights, modelConfig, runtime, {
      profile: canProfileGpu,
    });
    if (canProfileGpu && Number.isFinite(promptCondition.profile?.totalMs)) {
      gpuPrefillMs += promptCondition.profile.totalMs;
    }
    let negativeCondition = null;
    if (shouldUseUncond) {
      negativeCondition = await runTextEncodersForPrompt(negativeTokens, textWeights, modelConfig, runtime, {
        profile: canProfileGpu,
      });
      if (canProfileGpu && Number.isFinite(negativeCondition.profile?.totalMs)) {
        gpuPrefillMs += negativeCondition.profile.totalMs;
      }
    }
    const promptEnd = performance.now();

    if (runtime.swapper?.enabled && runtime.swapper?.evictTextEncoder) {
      this.releaseTextEncoderWeights();
    }

    const transformerWeights = await this.ensureTransformerWeights();
    const transformerConfig = modelConfig?.components?.transformer?.config || {};
    const transformerResolver = createSD3WeightResolver(transformerWeights, modelConfig);
    const hiddenSize = (transformerConfig.num_attention_heads ?? 0) * (transformerConfig.attention_head_dim ?? 0);
    const patchSize = transformerConfig.patch_size ?? 2;
    const timeEmbedWeight = transformerResolver.get('time_text_embed.timestep_embedder.linear_1.weight');
    const timeEmbedDim = timeEmbedWeight?.shape?.[1] ?? transformerConfig.time_embed_dim ?? 256;
    if (!Number.isFinite(hiddenSize) || hiddenSize <= 0) {
      throw new Error('Diffusion transformer config missing num_attention_heads/attention_head_dim.');
    }
    const prefillRecorder = canProfileGpu
      ? new CommandRecorder(getDevice(), 'diffusion_prefill', { profile: true })
      : null;
    const condContext = await projectContext(promptCondition.context, transformerWeights, modelConfig, runtime, {
      recorder: prefillRecorder,
    });
    const uncondContext = shouldUseUncond && negativeCondition
      ? await projectContext(negativeCondition.context, transformerWeights, modelConfig, runtime, {
          recorder: prefillRecorder,
        })
      : null;
    if (prefillRecorder) {
      prefillRecorder.submit();
      const timings = await prefillRecorder.resolveProfileTimings();
      const contextMs = sumProfileTimings(timings);
      if (Number.isFinite(contextMs)) {
        gpuPrefillMs += contextMs;
      }
    }

    const scheduler = buildScheduler(runtime.scheduler, steps);
    if (scheduler.type !== 'flowmatch_euler') {
      log.warn('Diffusion', `GPU pipeline tuned for flowmatch_euler; running "${scheduler.type}" may be inaccurate.`);
    }
    const latentScale = this.diffusionState.latentScale;
    const latentChannels = this.diffusionState.latentChannels;
    const { latents, latentWidth, latentHeight } = generateLatents(width, height, latentChannels, latentScale, seed);
    if (scheduler.sigmas?.length) {
      const sigma0 = scheduler.sigmas[0];
      for (let i = 0; i < latents.length; i++) {
        latents[i] *= sigma0;
      }
    }

    if (latentWidth % patchSize !== 0 || latentHeight % patchSize !== 0) {
      throw new Error(`Latent size ${latentWidth}x${latentHeight} must be divisible by patch size ${patchSize}.`);
    }

    let latentsTensor = await createLatentTensor(
      latents,
      [latentChannels, latentHeight, latentWidth],
      runtime
    );

    this._onProgress?.({
      stage: 'diffusion',
      message: `Denoising ${scheduler.steps} steps...`,
      progress: 0,
    });

    const decodeStart = performance.now();
    const latentSize = latentChannels * latentHeight * latentWidth;
    for (let i = 0; i < scheduler.steps; i++) {
      const timestep = scheduler.timesteps[i];
      const sigma = scheduler.sigmas[i];
      const sigmaNext = i + 1 < scheduler.steps ? scheduler.sigmas[i + 1] : 0;
      const delta = sigmaNext - sigma;
      const stepRecorder = canProfileGpu
        ? new CommandRecorder(getDevice(), `diffusion_step_${i}`, { profile: true })
        : null;
      const releaseStep = createRecorderReleaser(stepRecorder);
      const scale = stepRecorder
        ? (input, scalar, options) => recordScale(stepRecorder, input, scalar, options)
        : runScale;
      const residualAdd = stepRecorder
        ? (left, right, count, options) => recordResidualAdd(stepRecorder, left, right, count, options)
        : runResidualAdd;

      const timeCond = await buildTimestepEmbedding(timestep, transformerWeights, modelConfig, runtime, {
        dim: timeEmbedDim,
        recorder: stepRecorder,
      });
      const textCond = await buildTimeTextEmbedding(promptCondition.pooled, transformerWeights, modelConfig, runtime, {
        recorder: stepRecorder,
      });
      const timeTextCond = await combineTimeTextEmbeddings(timeCond, textCond, hiddenSize, {
        recorder: stepRecorder,
      });
      const condPred = await runSD3Transformer(latentsTensor, condContext, timeTextCond, transformerWeights, modelConfig, runtime, {
        recorder: stepRecorder,
      });
      releaseStep(timeTextCond.buffer);

      let pred = condPred;
      if (shouldUseUncond && uncondContext && negativeCondition) {
        const timeUncond = await buildTimestepEmbedding(timestep, transformerWeights, modelConfig, runtime, {
          dim: timeEmbedDim,
          recorder: stepRecorder,
        });
        const textUncond = await buildTimeTextEmbedding(negativeCondition.pooled, transformerWeights, modelConfig, runtime, {
          recorder: stepRecorder,
        });
        const timeTextUncond = await combineTimeTextEmbeddings(timeUncond, textUncond, hiddenSize, {
          recorder: stepRecorder,
        });
        const uncondPred = await runSD3Transformer(latentsTensor, uncondContext, timeTextUncond, transformerWeights, modelConfig, runtime, {
          recorder: stepRecorder,
        });
        releaseStep(timeTextUncond.buffer);
        pred = await applyGuidance(uncondPred, condPred, guidanceScale, latentSize, {
          recorder: stepRecorder,
          release: releaseStep,
        });
        releaseStep(uncondPred.buffer);
        releaseStep(condPred.buffer);
      }

      const scaled = await scale(pred, delta, { count: latentSize });
      const updated = await residualAdd(latentsTensor, scaled, latentSize, { useVec4: true });

      releaseStep(latentsTensor.buffer);
      releaseStep(scaled.buffer);
      releaseStep(pred.buffer);

      latentsTensor = createTensor(updated.buffer, updated.dtype, [latentChannels, latentHeight, latentWidth], 'sd3_latents');

      if (stepRecorder) {
        stepRecorder.submit();
        const timings = await stepRecorder.resolveProfileTimings();
        const stepMs = sumProfileTimings(timings);
        if (Number.isFinite(stepMs)) {
          gpuDenoiseMs += stepMs;
        }
      }

      if (i % 5 === 0 || i === scheduler.steps - 1) {
        this._onProgress?.({
          stage: 'diffusion',
          message: `Denoising ${i + 1}/${scheduler.steps}`,
          progress: (i + 1) / scheduler.steps,
        });
      }
    }
    const decodeEnd = performance.now();

    if (condContext?.buffer) releaseBuffer(condContext.buffer);
    if (uncondContext?.buffer) releaseBuffer(uncondContext.buffer);

    if (runtime.swapper?.enabled && runtime.swapper?.evictUnet) {
      this.releaseTransformerWeights();
    }

    const vaeStart = performance.now();
    const useGpuVae = runtime?.backend?.pipeline === 'gpu';
    if (useGpuVae) {
      await this.ensureVaeWeights();
    }
    const latentArray = await readTensorToFloat32(latentsTensor);
    releaseBuffer(latentsTensor.buffer);

    const vaeProfile = canProfileGpu ? {} : null;
    const pixels = await decodeLatents(latentArray, {
      width,
      height,
      latentWidth,
      latentHeight,
      latentChannels,
      latentScale,
      weights: useGpuVae ? this.vaeWeights : null,
      modelConfig,
      runtime,
      profile: vaeProfile,
    });
    const vaeEnd = performance.now();
    if (vaeProfile && Number.isFinite(vaeProfile.totalMs)) {
      gpuVaeMs = vaeProfile.totalMs;
    }

    const end = performance.now();
    const cpuPrefillMs = promptEnd - promptStart;
    const cpuDenoiseMs = decodeEnd - decodeStart;
    const cpuVaeMs = vaeEnd - vaeStart;
    const gpuTotalMs = canProfileGpu
      ? [gpuPrefillMs, gpuDenoiseMs, gpuVaeMs].reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0)
      : null;
    this.stats = {
      totalTimeMs: end - start,
      prefillTimeMs: cpuPrefillMs,
      prefillTokens: encoded.totalTokens,
      decodeTimeMs: cpuDenoiseMs,
      decodeTokens: scheduler.steps,
      vaeTimeMs: cpuVaeMs,
      gpu: canProfileGpu
        ? {
            available: true,
            totalMs: gpuTotalMs,
            prefillMs: gpuPrefillMs,
            denoiseMs: gpuDenoiseMs,
            vaeMs: gpuVaeMs,
          }
        : { available: false },
    };

    log.info('Diffusion', `Prompt encode: ${(promptEnd - promptStart).toFixed(0)}ms (${encoded.totalTokens} tokens)`);
    log.info('Diffusion', `Denoise: ${(decodeEnd - decodeStart).toFixed(0)}ms (${scheduler.steps} steps)`);
    log.info('Diffusion', `VAE decode: ${(vaeEnd - vaeStart).toFixed(0)}ms (${width}x${height})`);
    log.info('Diffusion', `Total: ${(end - start).toFixed(0)}ms`);
    trace.perf('Diffusion summary', {
      prefillMs: cpuPrefillMs,
      prefillTokens: encoded.totalTokens,
      denoiseMs: cpuDenoiseMs,
      steps: scheduler.steps,
      vaeMs: cpuVaeMs,
      totalMs: end - start,
      gpuPrefillMs: canProfileGpu ? gpuPrefillMs : null,
      gpuDenoiseMs: canProfileGpu ? gpuDenoiseMs : null,
      gpuVaeMs: canProfileGpu ? gpuVaeMs : null,
      gpuTotalMs: canProfileGpu ? gpuTotalMs : null,
      width,
      height,
    });

    return { width, height, pixels };
  }
}

export async function createDiffusionPipeline(manifest, contexts = {}) {
  const pipeline = new DiffusionPipeline();
  await pipeline.initialize(contexts);
  await pipeline.loadModel(manifest);
  return pipeline;
}

registerPipeline('diffusion', createDiffusionPipeline);
