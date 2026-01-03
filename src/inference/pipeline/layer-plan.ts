/**
 * Layer pipeline plan compiler.
 *
 * Converts JSON-configured step lists into normalized, validated plans
 * for execution inside processLayer.
 *
 * @module inference/pipeline/layer-plan
 */

import { log } from '../../debug/index.js';
import type {
  LayerPipelineSchema,
  LayerPipelineStepSchema,
  LayerPipelineOp,
  LayerPipelineNormWeight,
  ProbeStage,
} from '../../config/schema/index.js';

export interface CompiledLayerPipelineStep {
  op: LayerPipelineOp;
  src: string;
  dst: string;
  name?: string;
  weight?: LayerPipelineNormWeight;
  residual?: string | null;
  a?: string;
  b?: string;
  variant?: 'auto' | 'dense' | 'moe';
  skipInputNorm?: boolean;
  probeStage?: ProbeStage;
}

export interface LayerPipelineOverride {
  layers: number[];
  steps: CompiledLayerPipelineStep[];
}

export interface CompiledLayerPipeline {
  steps: CompiledLayerPipelineStep[];
  overrides: LayerPipelineOverride[];
  source: 'model' | 'runtime';
}

const DEFAULT_SLOT = 'state';

function normalizeSlot(name?: string): string {
  const value = (name ?? '').trim();
  return value.length > 0 ? value : DEFAULT_SLOT;
}

function requireName(value: string | undefined, label: string): string {
  const normalized = (value ?? '').trim();
  if (!normalized) {
    throw new Error(`Layer pipeline step "${label}" requires a non-empty name`);
  }
  return normalized;
}

function normalizeLayers(layers: number[], numLayers: number): number[] {
  const seen = new Set<number>();
  const result: number[] = [];
  for (const entry of layers) {
    const layer = Number(entry);
    if (!Number.isInteger(layer)) continue;
    if (layer < 0 || layer >= numLayers) continue;
    if (seen.has(layer)) continue;
    seen.add(layer);
    result.push(layer);
  }
  return result;
}

function compileStep(step: LayerPipelineStepSchema, index: number): CompiledLayerPipelineStep {
  const op = step.op;
  const src = normalizeSlot(step.src);
  const dst = normalizeSlot(step.dst);

  switch (op) {
    case 'save': {
      const name = requireName(step.name, `save@${index}`);
      return { op, src, dst, name, probeStage: step.probeStage };
    }
    case 'load': {
      const name = requireName(step.name, `load@${index}`);
      return { op, src, dst, name, probeStage: step.probeStage };
    }
    case 'attention':
      return {
        op,
        src,
        dst,
        residual: step.residual ?? null,
        skipInputNorm: step.skipInputNorm === true,
        probeStage: step.probeStage,
      };
    case 'rmsnorm': {
      if (!step.weight) {
        throw new Error(`Layer pipeline step "rmsnorm@${index}" requires weight`);
      }
      // Deprecation warning for post_attention -> post_attn
      if (step.weight === 'post_attention') {
        log.warn('Pipeline', `Step ${index}: "post_attention" is deprecated, use "post_attn"`);
      }
      return {
        op,
        src,
        dst,
        weight: step.weight,
        residual: step.residual ?? null,
        probeStage: step.probeStage,
      };
    }
    case 'ffn':
      return {
        op,
        src,
        dst,
        variant: step.variant ?? 'auto',
        probeStage: step.probeStage,
      };
    case 'residual_add':
      return {
        op,
        src,
        dst,
        a: normalizeSlot(step.a ?? DEFAULT_SLOT),
        b: normalizeSlot(step.b ?? 'residual'),
        probeStage: step.probeStage,
      };
    case 'noop':
      return { op, src, dst, probeStage: step.probeStage };
    default:
      throw new Error(`Unknown layer pipeline op "${op}" at step ${index}`);
  }
}

function compileSteps(steps: LayerPipelineStepSchema[], label: string): CompiledLayerPipelineStep[] {
  if (!Array.isArray(steps) || steps.length === 0) {
    throw new Error(`Layer pipeline "${label}" must define a non-empty steps array`);
  }
  return steps.map((step, index) => compileStep(step, index));
}

/**
 * Validate slot lifetimes: ensure all reads reference previously-defined slots.
 * Throws at compile time if a step reads an undefined slot.
 */
function validateSlotLifetimes(steps: CompiledLayerPipelineStep[], label: string): void {
  const defined = new Set<string>(['state']); // 'state' is always available

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];

    // Collect all slots this step reads
    const reads: string[] = [];
    if (step.src && step.src !== 'state') reads.push(step.src);
    if (step.op === 'load' && step.name) reads.push(step.name);
    if (step.op === 'residual_add') {
      if (step.a && step.a !== 'state') reads.push(step.a);
      if (step.b) reads.push(step.b);
    }
    if (step.residual) reads.push(step.residual);

    // Check each read references a defined slot
    for (const slot of reads) {
      if (!defined.has(slot)) {
        throw new Error(`Pipeline "${label}" step ${i} (${step.op}) reads undefined slot "${slot}"`);
      }
    }

    // Track writes (save creates named slot, dst updates slot)
    if (step.op === 'save' && step.name) defined.add(step.name);
    if (step.dst && step.dst !== 'state') defined.add(step.dst);
  }
}

export function compileLayerPipeline(
  plan: LayerPipelineSchema,
  numLayers: number
): Omit<CompiledLayerPipeline, 'source'> {
  const compiledSteps = compileSteps(plan.steps, 'default');
  validateSlotLifetimes(compiledSteps, 'default');

  const overrides: LayerPipelineOverride[] = [];

  for (const override of plan.overrides ?? []) {
    const layers = normalizeLayers(override.layers ?? [], numLayers);
    if (layers.length === 0) {
      log.warn('Pipeline', 'Layer pipeline override has no valid layers, skipping');
      continue;
    }
    const label = `override@${layers.join(',')}`;
    const compiledOverrideSteps = compileSteps(override.steps, label);
    validateSlotLifetimes(compiledOverrideSteps, label);
    overrides.push({ layers, steps: compiledOverrideSteps });
  }

  return { steps: compiledSteps, overrides };
}

export function resolveLayerPipeline(
  modelPlan: LayerPipelineSchema | null | undefined,
  runtimePlan: LayerPipelineSchema | null | undefined,
  numLayers: number
): CompiledLayerPipeline | null {
  const runtimeHasSteps = runtimePlan?.steps && runtimePlan.steps.length > 0;
  const modelHasSteps = modelPlan?.steps && modelPlan.steps.length > 0;

  if (runtimeHasSteps) {
    return { ...compileLayerPipeline(runtimePlan!, numLayers), source: 'runtime' };
  }
  if (modelHasSteps) {
    return { ...compileLayerPipeline(modelPlan!, numLayers), source: 'model' };
  }

  return null;
}

export function getLayerPlanSteps(plan: CompiledLayerPipeline, layerIdx: number): CompiledLayerPipelineStep[] {
  for (const override of plan.overrides) {
    if (override.layers.includes(layerIdx)) {
      return override.steps;
    }
  }
  return plan.steps;
}
