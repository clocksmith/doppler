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
  graphContext: { activationDtype: string; kvDtype: string }
): ResolvedTransforms;

/**
 * Resolve the finiteness fallback transform (widenToF32Activations when activation is f16).
 * Returns null when already f32 (no fallback available).
 */
export declare function resolveFinitenessFallbackTransform(
  activationDtype: string
): ExecutionGraphTransform | null;
