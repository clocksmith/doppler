/**
 * Kernel Plan Registry
 *
 * Resolves and stores the active kernel plan used for pipeline execution.
 *
 * @module config/kernel-plan
 */

import type {
  KernelPlanSchema,
  KernelVariantOverrideSchema,
  KernelVariantOverridesSchema,
  Q4KStrategy,
} from './schema/index.js';
import { log } from '../debug/index.js';

export type KernelPlanSource = 'model' | 'runtime' | 'merged' | 'auto' | 'none';

export interface KernelPlanLookup {
  operation: keyof KernelVariantOverridesSchema;
  phase?: 'prefill' | 'decode';
  role?: string;
}

let currentPlan: KernelPlanSchema | null = null;
let currentSource: KernelPlanSource = 'none';

export function getKernelPlan(): KernelPlanSchema | null {
  return currentPlan;
}

export function getKernelPlanSource(): KernelPlanSource {
  return currentSource;
}

export function getKernelPlanStrict(): boolean {
  return currentPlan?.strict ?? false;
}

export function getKernelPlanQ4KStrategy(): Q4KStrategy {
  return currentPlan?.q4kStrategy ?? 'auto';
}

export function setKernelPlan(plan: KernelPlanSchema | null, source: KernelPlanSource): void {
  currentPlan = plan;
  currentSource = source;
}

function mergeVariantOverrides(
  base: KernelVariantOverridesSchema | undefined,
  override: KernelVariantOverridesSchema | undefined
): KernelVariantOverridesSchema | undefined {
  if (!base && !override) return undefined;
  if (!base) return override;
  if (!override) return base;

  const result: KernelVariantOverridesSchema = { ...base };
  for (const [key, value] of Object.entries(override) as Array<[keyof KernelVariantOverridesSchema, KernelVariantOverrideSchema | undefined]>) {
    if (!value) continue;
    const baseValue = base[key];
    if (!baseValue) {
      result[key] = value;
      continue;
    }
    result[key] = {
      default: value.default ?? baseValue.default,
      prefill: value.prefill ?? baseValue.prefill,
      decode: value.decode ?? baseValue.decode,
      roles: { ...(baseValue.roles ?? {}), ...(value.roles ?? {}) },
    };
  }
  return result;
}

function mergeKernelPlan(
  base: KernelPlanSchema | null | undefined,
  override: KernelPlanSchema | null | undefined
): KernelPlanSchema | null {
  if (!base && !override) return null;
  if (!override) return base ?? null;
  if (!base) return override;

  if (override.mode === 'replace') {
    return override;
  }

  return {
    mode: override.mode ?? base.mode ?? 'patch',
    layerPipeline: override.layerPipeline !== undefined ? override.layerPipeline : base.layerPipeline,
    variants: mergeVariantOverrides(base.variants, override.variants),
    q4kStrategy: override.q4kStrategy ?? base.q4kStrategy,
    strict: override.strict ?? base.strict,
  };
}

export function resolveKernelPlan(
  modelPlan: KernelPlanSchema | null | undefined,
  runtimePlan: KernelPlanSchema | null | undefined
): { plan: KernelPlanSchema | null; source: KernelPlanSource } {
  if (!runtimePlan && !modelPlan) {
    return { plan: null, source: 'none' };
  }

  if (runtimePlan && runtimePlan.mode === 'replace') {
    return { plan: runtimePlan, source: 'runtime' };
  }

  if (runtimePlan && modelPlan) {
    return { plan: mergeKernelPlan(modelPlan, runtimePlan), source: 'merged' };
  }

  return { plan: runtimePlan ?? modelPlan ?? null, source: runtimePlan ? 'runtime' : 'model' };
}

export function resolveKernelVariant(
  override: KernelVariantOverrideSchema | undefined,
  lookup: Omit<KernelPlanLookup, 'operation'>
): string | null {
  if (!override) return null;
  if (lookup.role && override.roles?.[lookup.role]) {
    return override.roles[lookup.role] ?? null;
  }
  if (lookup.phase && override[lookup.phase]) {
    return override[lookup.phase] ?? null;
  }
  return override.default ?? null;
}

export function getKernelPlanVariant(lookup: KernelPlanLookup): string | null {
  const plan = currentPlan;
  if (!plan?.variants) return null;
  const override = plan.variants[lookup.operation];
  return resolveKernelVariant(override, lookup);
}

export function logKernelPlanSummary(label: string = 'KernelPlan'): void {
  if (!currentPlan) {
    log.debug(label, 'No kernel plan configured');
    return;
  }
  const summary = {
    source: currentSource,
    strict: currentPlan.strict ?? true,
    q4kStrategy: currentPlan.q4kStrategy ?? 'auto',
    hasLayerPipeline: currentPlan.layerPipeline != null,
    variantOverrides: Object.keys(currentPlan.variants ?? {}),
  };
  log.debug(label, `Resolved kernel plan: ${JSON.stringify(summary)}`);
}
