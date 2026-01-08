/**
 * test-harness.ts - Shared Inference Test Utilities
 *
 * Common utilities for inference testing and automation:
 * - Model discovery via /api/models
 * - URL parameter parsing for runtime overrides
 * - HTTP-based shard loading
 * - Pipeline initialization helpers
 *
 * Used by test-inference.html and potentially other test harnesses.
 *
 * @module inference/test-harness
 */

import { type KernelCapabilities } from '../gpu/device.js';
import { type RDRRManifest } from '../storage/rdrr-format.js';
import { type Pipeline } from './pipeline.js';
import type { RuntimeConfigSchema } from '../config/schema/index.js';
import type { KernelPathSchema } from '../config/schema/index.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Model info returned from /api/models
 */
export interface ModelInfo {
  id: string;
  name: string;
  path?: string;
  numLayers?: number;
  vocabSize?: number;
  quantization?: string;
  downloadSize?: number;
  architecture?: string;
}

/**
 * Runtime overrides parsed from URL parameters
 */
export interface RuntimeOverrides {
  debug?: boolean;
  /**
   * Kernel path for explicit kernel dispatch ordering.
   * Can be a preset ID (e.g., 'gemma2-q4k-fused') or inline KernelPathSchema.
   */
  kernelPath?: string | KernelPathSchema;
  runtimeConfig?: Partial<RuntimeConfigSchema>;
  /** Enable GPU timestamp profiling */
  profile?: boolean;
  /** Trace level: 'quick' | 'full' */
  trace?: string;
  /** Specific layers to debug checkpoint */
  debugLayers?: number[];
  /** Config inheritance chain for debugging (e.g., ['debug', 'default']) */
  configChain?: string[];
}

/**
 * Options for pipeline initialization
 */
export interface InferenceHarnessOptions {
  /** Base URL for model files (default: inferred from model URL) */
  baseUrl?: string;
  /** Runtime overrides for kernel selection */
  runtime?: RuntimeOverrides;
  /** Progress callback */
  onProgress?: (phase: string, progress: number, detail?: string) => void;
  /** Log function (default: debug log) */
  log?: (msg: string, level?: string) => void;
}

/**
 * Result of pipeline initialization
 */
export interface InitializeResult {
  pipeline: Pipeline;
  manifest: RDRRManifest;
  capabilities: KernelCapabilities;
}

// ============================================================================
// Model Discovery
// ============================================================================

/**
 * Discover available models from the /api/models endpoint.
 *
 * @param fallbackModels - Models to return if API fails
 * @returns Array of model info objects
 */
export declare function discoverModels(
  fallbackModels?: string[]
): Promise<ModelInfo[]>;

// ============================================================================
// URL Parameter Parsing
// ============================================================================

/**
 * Parse runtime overrides from URL query parameters.
 *
 * Supported parameters:
 * - debug: Enable debug mode
 *
 * @param searchParams - URLSearchParams to parse (default: window.location.search)
 * @returns RuntimeOverrides object
 */
export declare function parseRuntimeOverridesFromURL(
  searchParams?: URLSearchParams
): RuntimeOverrides;

// ============================================================================
// Shard Loading
// ============================================================================

/**
 * Create an HTTP-based shard loader for a model.
 *
 * @param baseUrl - Base URL for the model (e.g., http://localhost:8080/doppler/models/gemma-1b-q4)
 * @param manifest - Parsed model manifest
 * @param log - Optional logging function
 * @returns Async function that loads a shard by index
 */
export declare function createHttpShardLoader(
  baseUrl: string,
  manifest: RDRRManifest,
  log?: (msg: string, level?: string) => void
): (idx: number) => Promise<Uint8Array>;

// ============================================================================
// Pipeline Initialization
// ============================================================================

/**
 * Fetch and parse a model manifest from a URL.
 *
 * @param manifestUrl - URL to manifest.json
 * @returns Parsed manifest
 */
export declare function fetchManifest(manifestUrl: string): Promise<RDRRManifest>;

/**
 * Initialize the WebGPU device and return capabilities.
 *
 * @returns Kernel capabilities
 */
export declare function initializeDevice(): Promise<KernelCapabilities>;

/**
 * Initialize a complete inference pipeline from a model URL.
 *
 * This is a convenience function that handles:
 * 1. WebGPU device initialization
 * 2. Manifest fetching and parsing
 * 3. Pipeline creation with shard loading
 *
 * @param modelUrl - Base URL for the model directory
 * @param options - Initialization options
 * @returns Pipeline and associated info
 */
export declare function initializeInference(
  modelUrl: string,
  options?: InferenceHarnessOptions
): Promise<InitializeResult>;

// ============================================================================
// Test State (for Playwright automation)
// ============================================================================

/**
 * Standard test state interface for Playwright automation.
 */
export interface TestState {
  ready: boolean;
  loading: boolean;
  loaded: boolean;
  generating: boolean;
  done: boolean;
  output: string;
  tokens: string[];
  errors: string[];
  model: string | null;
}

/**
 * Create initial test state object.
 */
export declare function createTestState(): TestState;
