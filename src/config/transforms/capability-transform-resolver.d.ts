/**
 * Capability Transform Resolver
 *
 * Resolves GPU capabilities and platform info to an ordered chain of
 * execution graph transforms. Used by the runtime to adapt the
 * manifest execution graph to the current device.
 *
 * @module config/transforms/capability-transform-resolver
 */

import type { ExecutionGraphTransform, TransformContext } from './execution-graph-transforms.js';

export interface ResolvedTransforms {
  transforms: ExecutionGraphTransform[];
  names: string[];
  reason: string;
}

/**
 * Resolve GPU capabilities and platform info to a chain of execution graph transforms.
 */
export declare function resolveCapabilityTransforms(
  capabilities: TransformContext['capabilities'],
  platform: TransformContext['platform'],
  graphContext: {
    activationDtype: string;
    kvDtype: string;
    modelId?: string;
    layerTypes?: string[] | null;
    hasDensePrefillProjectionKernel?: boolean;
    hasQ4DecodeProjectionKernel?: boolean;
    hasQ4PrefillProjectionKernel?: boolean;
    hasAvailableQ4PrefillProjectionKernel?: boolean;
  }
): ResolvedTransforms;

/**
 * Resolve the explicit alternate-plan transform for finiteness handling
 * (widenToF32Activations when activation is f16).
 * Returns null when already f32 (no alternate plan available).
 */
export declare function resolveFinitenessFallbackTransform(
  graphContext: { activationDtype: string; kvDtype: string; modelId?: string; layerTypes?: string[] | null }
): { transform: ExecutionGraphTransform; name: string } | null;
