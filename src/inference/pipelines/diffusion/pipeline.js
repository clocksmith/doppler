import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { log, trace } from '../../../debug/index.js';
import { registerPipeline } from '../registry.js';
import { applyPipelineContexts, restorePipelineContexts } from '../context.js';
import { createInitializedPipeline } from '../factory.js';
import { createRng, sampleNormal } from '../rng.js';
import { initializeDiffusion } from './init.js';
import { loadDiffusionTokenizers, encodePrompt } from './text-encoder.js';
import {
  runTextEncodersForPrompt,
  buildTimeTextEmbedding,
  buildTimestepEmbedding,
  combineTimeTextEmbeddings,
  projectContext,
  assertClipHiddenActivationSupported,
} from './text-encoder-gpu.js';
import { buildScheduler, stepScmScheduler } from './scheduler.js';
import { decodeLatents } from './vae.js';
import { createDiffusionWeightLoader } from './weights.js';
import { runSD3Transformer } from './sd3-transformer.js';
import { createSD3WeightResolver } from './sd3-weights.js';
import { createTensor, dtypeBytes } from '../../../gpu/tensor.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { CommandRecorder } from '../../../gpu/command-recorder.js';
import { castF32ToF16 } from '../../../gpu/kernels/cast.js';
import { runResidualAdd, runScale, recordResidualAdd, recordScale } from '../../../gpu/kernels/index.js';
import { f16ToF32 } from '../../../loader/dtype-utils.js';

const SUPPORTED_DIFFUSION_BACKEND_PIPELINES = new Set(['gpu']);
const DEFAULT_TIME_EMBED_DIM = 256;
const SD3_TEXT_ENCODER_KEYS = ['text_encoder', 'text_encoder_2', 'text_encoder_3'];

function createRandomSeed() {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    return values[0] >>> 0;
  }
  return Date.now() >>> 0;
}

function generateLatents(width, height, channels, latentScale, seed) {
  const latentWidth = Math.max(1, Math.floor(width / latentScale));
  const latentHeight = Math.max(1, Math.floor(height / latentScale));
  const size = latentWidth * latentHeight * channels;
  const latents = new Float32Array(size);
  const rand = createRng(seed ?? createRandomSeed());
  for (let i = 0; i < size; i++) {
    latents[i] = sampleNormal(rand);
  }
  return { latents, latentWidth, latentHeight };
}

function generateNoiseVector(size, seed) {
  if (!Number.isFinite(size) || size <= 0) {
    throw new Error(`generateNoiseVector requires a positive size, got ${size}.`);
  }
  const out = new Float32Array(size);
  const rand = createRng(seed ?? createRandomSeed());
  for (let i = 0; i < size; i++) {
    out[i] = sampleNormal(rand);
  }
  return out;
}

function extractTokenSet(tokensByEncoder, key) {
  const output = {};
  for (const [name, entry] of Object.entries(tokensByEncoder || {})) {
    const tokens = entry?.[key];
    output[name] = Array.isArray(tokens) ? tokens : [];
  }
  return output;
}

function resolveDiffusionLayout(modelConfig) {
  return modelConfig?.layout ?? 'sd3';
}

function getTextEncoderKeysForLayout() {
  return SD3_TEXT_ENCODER_KEYS;
}

function assertLayoutTextEncoderContract(layout, modelConfig, tokenizers) {
  const requiredKeys = getTextEncoderKeysForLayout(layout);
  for (const key of requiredKeys) {
    if (!modelConfig?.components?.[key]) {
      throw new Error(`Diffusion GPU pipeline requires component "${key}" for layout "${layout}".`);
    }
    if (!tokenizers?.[key]) {
      throw new Error(`Diffusion GPU pipeline requires tokenizer "${key}" for layout "${layout}".`);
    }
  }
}

function buildTokenizerMaxLengths(layout, runtime) {
  const maxLength = runtime?.textEncoder?.maxLength;
  if (!Number.isFinite(maxLength) || maxLength <= 0) {
    throw new Error('Diffusion runtime requires runtime.textEncoder.maxLength.');
  }
  const t5MaxLength = runtime?.textEncoder?.t5MaxLength ?? maxLength;
  if (!Number.isFinite(t5MaxLength) || t5MaxLength <= 0) {
    throw new Error('Diffusion runtime requires runtime.textEncoder.t5MaxLength (or runtime.textEncoder.maxLength).');
  }
  return {
    text_encoder: maxLength,
    text_encoder_2: maxLength,
    text_encoder_3: t5MaxLength,
  };
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

async function applySchedulerStep(latentsTensor, scheduler, stepIndex, timestep, predictionTensor, runtime, options = {}) {
  if (scheduler.type === 'flowmatch_euler') {
    const sigma = scheduler.sigmas[stepIndex];
    const sigmaNext = stepIndex + 1 < scheduler.steps ? scheduler.sigmas[stepIndex + 1] : 0;
    const delta = sigmaNext - sigma;
    const latentSize = getTensorSize(latentsTensor.shape);
    const scale = options.scale ?? runScale;
    const residualAdd = options.residualAdd ?? runResidualAdd;
    const release = options.release ?? releaseBuffer;

    const scaled = await scale(predictionTensor, delta, { count: latentSize });
    const updated = await residualAdd(latentsTensor, scaled, latentSize, { useVec4: true });

    release(latentsTensor.buffer);
    release(scaled.buffer);
    release(predictionTensor.buffer);

    return createTensor(updated.buffer, updated.dtype, [...latentsTensor.shape], 'diffusion_latents');
  }

  if (scheduler.type === 'scm') {
    const sample = await readTensorToFloat32(latentsTensor);
    const modelOutput = await readTensorToFloat32(predictionTensor);
    releaseBuffer(predictionTensor.buffer);
    releaseBuffer(latentsTensor.buffer);

    const isFinalStep = stepIndex + 1 >= scheduler.timesteps.length - 1;
    const noise = isFinalStep
      ? null
      : generateNoiseVector(
          sample.length,
          (options.seedBase ?? createRandomSeed()) + stepIndex + 1
        );
    const step = stepScmScheduler(scheduler, modelOutput, timestep, sample, stepIndex, noise);
    return createLatentTensor(step.prevSample, [...latentsTensor.shape], runtime);
  }

  throw new Error(`Unsupported diffusion scheduler.type "${scheduler.type}".`);
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
  weightLoader = null;
  vaeWeights = null;
  textEncoderWeights = null;
  transformerWeights = null;

  async initialize(contexts = {}) {
    const { runtimeConfig } = applyPipelineContexts(this, contexts);
    this.runtimeConfig = runtimeConfig;
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
    if (!SUPPORTED_DIFFUSION_BACKEND_PIPELINES.has(pipelineMode)) {
      throw new Error(
        `Unsupported diffusion backend.pipeline "${pipelineMode}". ` +
        'Expected: gpu.'
      );
    }
    log.info('Diffusion', 'GPU diffusion pipeline enabled.');
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
    this.weightLoader = null;
    this.vaeWeights = null;
    this.textEncoderWeights = null;
    this.transformerWeights = null;
    restorePipelineContexts(this);
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

    const layout = resolveDiffusionLayout(this.diffusionState?.modelConfig);
    const requiredKeys = getTextEncoderKeysForLayout(layout);
    const weights = {};
    for (const key of requiredKeys) {
      weights[key] = await this.weightLoader.loadComponentWeights(key);
    }

    this.textEncoderWeights = {
      text_encoder: weights.text_encoder ?? null,
      text_encoder_2: weights.text_encoder_2 ?? null,
      text_encoder_3: weights.text_encoder_3 ?? null,
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
    throw new Error(
      `Unsupported diffusion backend.pipeline "${pipelineMode}". ` +
      'Expected: gpu.'
    );
  }

  async generateCPU(request = {}) {
    throw new Error(
      'Diffusion CPU pipeline is not supported. ' +
      'Set runtime.inference.diffusion.backend.pipeline="gpu".'
    );
  }

  async generateGPU(request = {}) {
    const start = performance.now();
    const runtime = this.diffusionState.runtime;
    const modelConfig = this.diffusionState.modelConfig;
    const layout = resolveDiffusionLayout(modelConfig);
    const tokenizerMaxLengths = buildTokenizerMaxLengths(layout, runtime);

    const defaultWidth = runtime.latent.width;
    const defaultHeight = runtime.latent.height;
    const width = Math.floor(Number.isFinite(request.width) && request.width > 0 ? request.width : defaultWidth);
    const height = Math.floor(Number.isFinite(request.height) && request.height > 0 ? request.height : defaultHeight);
    let steps = Math.floor(Number.isFinite(request.steps) && request.steps > 0 ? request.steps : runtime.scheduler.numSteps);
    const guidanceScale = Number.isFinite(request.guidanceScale) && request.guidanceScale > 0
      ? request.guidanceScale
      : runtime.scheduler.guidanceScale;
    const seed = Number.isFinite(request.seed) ? Math.floor(request.seed) : createRandomSeed();
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
    const maxAllowedSteps = runtime.scheduler.maxSteps ?? 1000;
    const minAllowedSteps = runtime.scheduler.minSteps ?? 1;
    if (steps > maxAllowedSteps) {
      log.warn('Diffusion', `Requested ${steps} steps exceeds maximum ${maxAllowedSteps}. Clamping to ${maxAllowedSteps}.`);
      steps = maxAllowedSteps;
    }
    if (steps < minAllowedSteps) {
      log.warn('Diffusion', `Requested ${steps} steps below minimum ${minAllowedSteps}. Clamping to ${minAllowedSteps}.`);
      steps = minAllowedSteps;
    }

    if (!modelConfig?.components?.transformer) {
      throw new Error('Diffusion GPU pipeline requires transformer component config.');
    }
    assertLayoutTextEncoderContract(layout, modelConfig, this.tokenizers);
    if (layout === 'sd3') {
      assertClipHiddenActivationSupported(modelConfig?.components?.text_encoder?.config || {});
    }

    const promptStart = performance.now();
    const encoded = encodePrompt(
      { prompt: request.prompt ?? '', negativePrompt: request.negativePrompt ?? '' },
      this.tokenizers || {},
      {
        maxLengthByTokenizer: tokenizerMaxLengths,
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
    const timeEmbedDim = timeEmbedWeight?.shape?.[1] ?? transformerConfig.time_embed_dim ?? DEFAULT_TIME_EMBED_DIM;
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

      const condPred = await (async () => {
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
        const output = await runSD3Transformer(latentsTensor, condContext, timeTextCond, transformerWeights, modelConfig, runtime, {
          recorder: stepRecorder,
        });
        releaseStep(timeTextCond.buffer);
        return output;
      })();

      let pred = condPred;
      if (shouldUseUncond && uncondContext && negativeCondition) {
        const uncondPred = await (async () => {
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
          const output = await runSD3Transformer(latentsTensor, uncondContext, timeTextUncond, transformerWeights, modelConfig, runtime, {
            recorder: stepRecorder,
          });
          releaseStep(timeTextUncond.buffer);
          return output;
        })();
        pred = await applyGuidance(uncondPred, condPred, guidanceScale, latentSize, {
          recorder: stepRecorder,
          release: releaseStep,
        });
        releaseStep(uncondPred.buffer);
        releaseStep(condPred.buffer);
      }

      latentsTensor = await applySchedulerStep(
        latentsTensor,
        scheduler,
        i,
        timestep,
        pred,
        runtime,
        {
          scale,
          residualAdd,
          release: releaseStep,
          seedBase: seed,
        }
      );

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
  return createInitializedPipeline(DiffusionPipeline, manifest, contexts);
}

registerPipeline('diffusion', createDiffusionPipeline);
