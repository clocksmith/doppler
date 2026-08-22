/**
 * Doppler Forge Compiler Pipeline Stages
 *
 * @module converter/forge-stages
 */

import { createModelIR, hashModelIR, validateModelIR } from '../config/model-ir.js';
import { createTargetPlan, hashTargetPlan, validateTargetPlan } from '../config/target-plan.js';
import { sha256Hex } from '../utils/sha256.js';

export const FORGE_PIPELINE_VERSION = '1.0.0';

/**
 * Stage 1: Inspect
 * Reads raw model metadata and file paths into a normalized intake record.
 */
export async function stageInspect(input) {
  const { modelDir, manifest = null, config = null } = input;
  return {
    stage: 'inspect',
    ok: true,
    data: {
      modelDir,
      hasManifest: Boolean(manifest),
      hasConfig: Boolean(config),
      sourceManifest: manifest,
      sourceConfig: config,
    },
  };
}

/**
 * Stage 2 & 3: Analyze
 * Extracts hardware-agnostic semantic ModelIR from source facts.
 */
export function stageAnalyze(intakeData) {
  const manifest = intakeData.sourceManifest || {};
  const config = intakeData.sourceConfig || {};

  const modelId = manifest.modelId || config.modelId || 'unnamed-model';
  const architecture = manifest.modelType || config.architecture || 'transformer';
  const hiddenSize = manifest.hiddenSize || config.hiddenSize || 2048;
  const numLayers = manifest.numLayers || config.numLayers || 18;
  const vocabSize = manifest.vocabSize || config.vocabSize || 32000;

  const ir = createModelIR({
    modelId,
    architecture,
    hiddenSize,
    numLayers,
    vocabSize,
    attentionGeometry: {
      numHeads: manifest.numHeads || config.numHeads || 16,
      numKvHeads: manifest.numKvHeads || config.numKvHeads || 4,
      headDim: Math.floor(hiddenSize / (manifest.numHeads || 16)),
    },
    normalization: {
      type: manifest.normType || 'rmsnorm',
      eps: manifest.normEps || 1e-6,
    },
    phases: ['prefill', 'decode'],
  });

  const irHash = hashModelIR(ir);
  return {
    stage: 'analyze',
    ok: true,
    modelIR: ir,
    modelIRHash: irHash,
  };
}

/**
 * Stage 4 & 5: Lower & Specialize
 * Lowers ModelIR into a set of discrete, specialized TargetPlans (e.g. f16-subgroups, f16, f32-safe).
 */
export function stageSpecialize(modelIR, kernelModules = []) {
  const validation = validateModelIR(modelIR);
  if (!validation.ok) {
    throw new Error(`stageSpecialize requires valid ModelIR: ${validation.errors.join('; ')}`);
  }

  // 1. High performance target: webgpu-f16-subgroups
  const f16SubgroupsPlan = createTargetPlan({
    targetId: 'webgpu-f16-subgroups',
    modelId: modelIR.modelId,
    capabilityPredicate: { requiresF16: true, requiresSubgroups: true, minBufferSize: 128 * 1024 * 1024 },
    dtypes: { activation: 'f16-subgroups', kv: 'f16', weight: 'q4k' },
    kernelClosure: kernelModules,
    memoryLayout: { kvCacheLayout: 'paged', estimatedPeakBytes: modelIR.hiddenSize * modelIR.numLayers * 1024 },
  });

  // 2. Standard mobile / laptop target: webgpu-f16
  const f16StandardPlan = createTargetPlan({
    targetId: 'webgpu-f16',
    modelId: modelIR.modelId,
    capabilityPredicate: { requiresF16: true, requiresSubgroups: false, minBufferSize: 64 * 1024 * 1024 },
    dtypes: { activation: 'f16', kv: 'f16', weight: 'q4k' },
    kernelClosure: kernelModules,
    memoryLayout: { kvCacheLayout: 'contiguous', estimatedPeakBytes: modelIR.hiddenSize * modelIR.numLayers * 1024 },
  });

  // 3. Fallback safe target: webgpu-f32-safe
  const f32SafePlan = createTargetPlan({
    targetId: 'webgpu-f32-safe',
    modelId: modelIR.modelId,
    capabilityPredicate: { requiresF16: false, requiresSubgroups: false, minBufferSize: 32 * 1024 * 1024 },
    dtypes: { activation: 'f32', kv: 'f32', weight: 'f32' },
    kernelClosure: kernelModules,
    memoryLayout: { kvCacheLayout: 'contiguous', estimatedPeakBytes: modelIR.hiddenSize * modelIR.numLayers * 2048 },
  });

  const targetPlans = [f16SubgroupsPlan, f16StandardPlan, f32SafePlan];
  const targetPlanHashes = targetPlans.map((p) => hashTargetPlan(p));

  return {
    stage: 'specialize',
    ok: true,
    targetPlans,
    targetPlanHashes,
  };
}

/**
 * Stage 6: Package
 * Packages ModelIR, TargetPlans, WGSL modules, and artifact descriptors into an immutable Doppler Pack v2.
 */
export function stagePackage(params) {
  const { modelIR, targetPlans = [], wgslModules = [], artifacts = [], packId = null } = params;
  const validation = validateModelIR(modelIR);
  if (!validation.ok) {
    throw new Error(`stagePackage requires valid ModelIR: ${validation.errors.join('; ')}`);
  }

  const generatedPackId = packId || `${modelIR.modelId}-pack-v2-${Date.now().toString(16)}`;
  const pack = {
    schema: 'doppler.pack/v2',
    schemaVersion: 2,
    packId: generatedPackId,
    modelId: modelIR.modelId,
    createdAtUtc: new Date().toISOString(),
    modelIR,
    targetPlans,
    wgslModules,
    artifacts,
    signature: null,
  };

  return {
    stage: 'package',
    ok: true,
    pack,
    packId: generatedPackId,
  };
}
