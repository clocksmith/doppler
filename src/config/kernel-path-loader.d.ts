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
 * Auto-select kernel path based on model quantization and capabilities.
 *
 * Selection priority:
 * - F16/BF16 models: use f16-native
 * - Q4K with subgroups: use fused path (best throughput)
 * - Q4K with F16 support: use dequant-f16 (balanced)
 * - Q4K fallback: use dequant-f32 (max compatibility)
 */
export function autoSelectKernelPath(
  quantization: string | null,
  modelFamily: string,
  capabilities?: { hasSubgroups?: boolean; hasF16?: boolean }
): KernelPathSchema;

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
export type KernelPathSource = 'runtime' | 'config' | 'model' | 'manifest' | 'auto' | 'none';

export function getKernelPathMatmulVariant(
  role: string | undefined,
  phase: KernelPathPhase,
  layerIndex?: number
): string | null;

export function getKernelPathAttentionVariant(
  phase: KernelPathPhase,
  layerIndex?: number
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
 * Check if the active kernel path uses fused Q4K matmul.
 * Returns false if no kernel path is set (auto-selection will apply).
 */
export function isActiveKernelPathFusedQ4K(): boolean;

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
