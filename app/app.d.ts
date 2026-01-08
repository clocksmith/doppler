/**
 * app.d.ts - DOPPLER Application Controller Type Declarations
 *
 * Main application that wires together all components and the DOPPLER inference pipeline.
 *
 * @module app/app
 */

import { ModelSelector, ModelInfo, ModelSources } from './model-selector.js';
import { ChatUI } from './chat-ui.js';
import { ProgressUI, SourceType } from './progress-ui.js';
import { QuickStartUI } from './quickstart-ui.js';
import { Pipeline } from '../src/inference/pipeline.js';
import { MemoryCapabilities } from '../src/memory/capability.js';
import type { KernelPathRef } from '../src/config/schema/index.js';

/**
 * Remote model definition
 */
export interface RemoteModel {
  id: string;
  name: string;
  size: string;
  quantization: string;
  downloadSize: number;
  url: string;
  source: string;
  downloaded?: boolean;
  architecture?: string;
}

/**
 * Capabilities state
 */
export interface Capabilities {
  webgpu: boolean;
  f16: boolean;
  subgroups: boolean;
  memory64: boolean;
}

/**
 * Stats DOM elements
 */
export interface StatsElements {
  tps: HTMLElement | null;
  memory: HTMLElement | null;
  gpu: HTMLElement | null;
  kv: HTMLElement | null;
}

/**
 * GPU info DOM elements
 */
export interface GPUElements {
  device: HTMLElement | null;
  vram: HTMLElement | null;
  vramLabel: HTMLElement | null;
  ram: HTMLElement | null;
  ramRow: HTMLElement | null;
  features: HTMLElement | null;
  unifiedNote: HTMLElement | null;
}

/**
 * Memory bar DOM elements
 */
export interface MemoryElements {
  heapBar: HTMLElement | null;
  heapValue: HTMLElement | null;
  gpuBar: HTMLElement | null;
  gpuValue: HTMLElement | null;
  kvBar: HTMLElement | null;
  kvValue: HTMLElement | null;
  opfsBar: HTMLElement | null;
  opfsValue: HTMLElement | null;
  headroomBar: HTMLElement | null;
  headroomValue: HTMLElement | null;
  heapStackedBar: HTMLElement | null;
  gpuStackedBar: HTMLElement | null;
  totalValue: HTMLElement | null;
}

/**
 * Registered model with sources
 */
export interface RegisteredModel extends ModelInfo {
  key: string;
  sources: ModelSources;
}

/**
 * Server model from API
 */
export interface ServerModel {
  name: string;
  path: string;
  size?: string;
  numLayers?: number;
  vocabSize?: number;
  quantization?: string;
  downloadSize?: number;
  architecture?: string;
}

/**
 * Main Demo Application
 */
export declare class DopplerDemo {
  /**
   * Initialize the application
   */
  init(): Promise<void>;

  /**
   * Select and load a model (run it)
   */
  selectModel(
    modelOrKey: RegisteredModel | string,
    opts?: { preferredSource?: string }
  ): Promise<void>;

  /**
   * Download/cache a model to browser storage
   */
  downloadModel(
    model: RegisteredModel,
    opts?: { runAfter?: boolean }
  ): Promise<void>;

  /**
   * Delete a model from browser cache
   */
  deleteModel(model: RegisteredModel): Promise<void>;

  /**
   * Send a chat message and generate response
   */
  chat(message: string): Promise<void>;

  /**
   * Stop current generation
   */
  stopGeneration(): void;

  /**
   * Clear conversation history
   */
  clearConversation(): void;

  /**
   * Get current status
   */
  getStatus(): {
    model: string | null;
    modelName: string | null;
    isGenerating: boolean;
    capabilities: Capabilities;
    memory: unknown;
    gpu: unknown;
  };

  /**
   * Start quick-start flow for a model
   */
  startQuickStart(modelId: string): Promise<void>;
}

export default DopplerDemo;
