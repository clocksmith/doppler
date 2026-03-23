/**
 * Kernel Path Loader
 *
 * Utility functions for kernel path objects and active kernel path singleton.
 * Registry loading removed in Phase 3 — kernel paths are now inline objects
 * resolved by execution graph transforms.
 *
 * @module config/kernel-path-loader
 */

import type {
  KernelPathSchema,
  KernelPathRef,
  KernelStepSchema,
} from './schema/kernel-path.schema.js';

/**
 * Return activation dtype required by a kernel path.
 * Returns null when the path does not specify an activation dtype.
 */
export function getKernelPathActivationDtype(
  path: KernelPathSchema | null
): string | null;

/**
 * Get the output dtype required by a kernel path.
 * Returns null when the path does not override output dtype.
 */
export function getKernelPathOutputDtype(
  path: KernelPathSchema | null
): string | null;

/**
 * Get the KV dtype for a kernel path.
 * Falls back to activationDtype if kvDtype is not set.
 */
export function getKernelPathKVDtype(
  path: KernelPathSchema | null
): string | null;

/**
 * Resolve a kernel path reference to a full schema object.
 * After registry removal (Phase 3), only object refs are supported.
 * String-based registry lookups throw an error.
 */
export function resolveKernelPath(ref: KernelPathRef): KernelPathSchema;

/**
 * Resolve layer index template in weight references.
 * Replaces {L} with the actual layer index.
 */
export function resolveWeightRef(template: string, layerIndex: number): string;

/**
 * Get steps for a specific layer, applying any overrides.
 */
export function getLayerSteps(
  path: KernelPathSchema,
  layerIndex: number,
  phase: 'prefill' | 'decode'
): KernelStepSchema[];

/**
 * Validate a kernel path schema.
 */
export function validateKernelPath(path: KernelPathSchema): string[];

export type KernelPathPhase = 'prefill' | 'decode';
export type KernelPathSection = 'layer' | 'preLayer' | 'postLayer' | 'sampling';
export type KernelPathSource = 'runtime' | 'config' | 'model' | 'manifest' | 'none';
export interface KernelPathPolicy {
  mode: 'locked' | 'capability-aware';
  sourceScope: KernelPathSource[];
  allowSources?: KernelPathSource[];
  onIncompatible: 'error' | 'remap';
}

export function getKernelPathMatmulVariant(
  role: string | undefined,
  phase: KernelPathPhase,
  layerIndex?: number,
  path?: KernelPathSchema | null
): string | null;

export function getKernelPathMatmulConstants(
  role: string | undefined,
  phase: KernelPathPhase,
  layerIndex?: number,
  path?: KernelPathSchema | null
): Record<string, number | boolean> | null;

export function getKernelPathAttentionVariant(
  phase: KernelPathPhase,
  layerIndex?: number,
  path?: KernelPathSchema | null
): string | null;

/**
 * Set the active kernel path for the current pipeline.
 * Called by Pipeline when resolving kernel path.
 */
export function setActiveKernelPath(
  path: KernelPathSchema | null,
  source?: KernelPathSource,
  policy?: KernelPathPolicy | null
): void;

/**
 * Get the active kernel path.
 */
export function getActiveKernelPath(): KernelPathSchema | null;

export function getActiveKernelPathSource(): KernelPathSource;
export function getActiveKernelPathPolicy(): KernelPathPolicy;

export function getKernelPathStrict(): boolean;

/**
 * Check if a kernel path uses fused Q4K matmul.
 */
export function isKernelPathFusedQ4K(path?: KernelPathSchema | null): boolean;

/**
 * Check if a kernel path requires matmul weights to stay in F32.
 */
export function kernelPathRequiresF32MatmulWeights(path?: KernelPathSchema | null): boolean;

/**
 * Check if the active kernel path uses fused Q4K matmul.
 */
export function isActiveKernelPathFusedQ4K(): boolean;

/**
 * Check if a kernel path uses dequant (non-fused) Q4K matmul.
 */
export function isKernelPathDequant(path?: KernelPathSchema | null): boolean;

/**
 * Check if the active kernel path uses dequant (non-fused) Q4K matmul.
 */
export function isActiveKernelPathDequant(): boolean;

/**
 * Format kernel path for logging.
 */
export function formatKernelPath(path: KernelPathSchema): string;

/**
 * Get summary statistics for a kernel path.
 */
export function getKernelPathStats(path: KernelPathSchema): {
  decodeSteps: number;
  prefillSteps: number;
  uniqueKernels: number;
  hasLayerOverrides: boolean;
};
