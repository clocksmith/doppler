/**
 * test-harness.ts - Shared Inference Test Utilities
 *
 * Common utilities for inference testing and automation:
 * - Model discovery via catalog.json
 * - URL parameter parsing for runtime config
 * - HTTP-based shard loading
 * - Pipeline initialization helpers
 *
 * Used by tests/harness.html (explicit mode/workload/modelId page context) and other test harnesses.
 *
 * @module inference/test-harness
 */

import { type KernelCapabilities } from '../gpu/device.js';
import { type RDRRManifest } from '../formats/rdrr/index.js';
import { type Pipeline } from './pipelines/text.js';
import type { RuntimeConfigSchema } from '../config/schema/index.js';
import type { PipelineStorageContext } from './pipelines/text/init.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Model info returned from catalog.json
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
  runtimeConfig?: Partial<RuntimeConfigSchema>;
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
  /** Explicit source loading mode from the command/suite request */
  loadMode?: 'opfs' | 'http' | 'memory' | 'file' | null;
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
 * Discover available models from the catalog.json endpoint.
 *
 * @param fallbackModels - Explicit fallback models to use when catalog fetch is unavailable
 * @returns Array of model info objects
 */
export declare function discoverModels(
  fallbackModels?: string[]
): Promise<ModelInfo[]>;

// ============================================================================
// URL Parameter Parsing
// ============================================================================

/**
 * Parse runtime config from URL query parameters.
 *
 * Supported parameters:
 * - runtimeConfig: JSON-encoded runtime config
 * - configChain: JSON-encoded config chain (for debugging)
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
 * @param baseUrl - Base URL for the model (e.g., http://localhost:8080/models/gemma-1b-q4)
 * @param manifest - Parsed model manifest
 * @param log - Optional logging function
 * @returns Async function that loads a shard by index
 */
export declare function createHarnessShardStorageContext(
  modelUrl: string,
  manifest: RDRRManifest,
  log?: (msg: string, level?: string) => void,
  options?: {
    loadMode?: 'opfs' | 'http' | 'memory' | 'file' | null;
  }
): PipelineStorageContext;

// ============================================================================
// Pipeline Initialization
// ============================================================================

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
// Test State (for browser automation)
// ============================================================================

/**
 * Standard test state interface for browser automation.
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
