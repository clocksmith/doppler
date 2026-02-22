/**
 * Kernel Path Loader
 *
 * Loads and resolves kernel path configurations.
 *
 * @module config/kernel-path-loader
 */

import type {
  KernelPathSchema,
  KernelPathRef,
  KernelStepSchema,
} from './schema/kernel-path.schema.js';

/**
 * Get a kernel path by ID.
 */
export function getKernelPath(id: string): KernelPathSchema | null;

/**
 * List all available kernel path IDs.
 */
export function listKernelPaths(): string[];

/**
 * Resolve a kernel path reference to a full schema.
 */
export function resolveKernelPath(ref: KernelPathRef): KernelPathSchema;

/**
 * Return activation dtype required by a kernel path.
 * Returns null when the path does not specify an activation dtype.
 */
export function getKernelPathActivationDtype(
  path: KernelPathSchema | null
): string | null;

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
export function setActiveKernelPath(path: KernelPathSchema | null, source?: KernelPathSource): void;

/**
 * Get the active kernel path.
 */
export function getActiveKernelPath(): KernelPathSchema | null;

export function getActiveKernelPathSource(): KernelPathSource;

export function getKernelPathStrict(): boolean;

/**
 * Check if a kernel path uses fused Q4K matmul.
 */
export function isKernelPathFusedQ4K(path?: KernelPathSchema | null): boolean;

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
