/**
 * model-selector.d.ts - Model Selection Component Type Declarations
 * Agent-D | Phase 2 | app/
 *
 * Handles model list display, download progress, and selection.
 *
 * @module app/model-selector
 */

/**
 * Model source locations
 */
export interface ModelSources {
  server?: { id: string; url: string };
  browser?: { id: string; url?: string };
  remote?: { id: string; url: string };
}

/**
 * Model information
 */
export interface ModelInfo {
  /** Unique model key */
  key: string;
  /** Display name */
  name: string;
  /** Model size (e.g., "1.1B") */
  size?: string;
  /** Model architecture */
  architecture?: string;
  /** Quantization format (e.g., "Q4_K_M") */
  quantization?: string;
  /** Download size in bytes */
  downloadSize?: number;
  /** Download progress (0-100) */
  downloadProgress?: number;
  /** Available sources */
  sources?: ModelSources;
  /** Whether Quick Start is available for this model (CDN with preflight checks) */
  quickStartAvailable?: boolean;
}

/**
 * Model selector callback functions
 */
export interface ModelSelectorCallbacks {
  /** Called when model is selected */
  onSelect?: (model: ModelInfo, opts?: { preferredSource?: string }) => void;
  /** Called when download is requested */
  onDownload?: (model: ModelInfo, opts?: { runAfter?: boolean }) => void;
  /** Called when delete is requested */
  onDelete?: (model: ModelInfo) => void;
  /** Called when Quick Start is requested (CDN download with preflight checks) */
  onQuickStart?: (model: ModelInfo) => void;
}

/**
 * Model selector class for managing model list UI
 */
export declare class ModelSelector {
  /**
   * @param container - Container element for model list
   * @param callbacks - Event callbacks
   */
  constructor(container: HTMLElement, callbacks?: ModelSelectorCallbacks);

  /**
   * Set available models
   */
  setModels(models: ModelInfo[]): void;

  /**
   * Update a single model's info
   */
  updateModel(modelKey: string, updates: Partial<ModelInfo>): void;

  /**
   * Set download progress for a model
   */
  setDownloadProgress(modelKey: string, progress: number): void;

  /**
   * Mark a model as downloaded (triggers refresh)
   */
  setDownloaded(modelKey: string): void;

  /**
   * Set the active (running) model
   */
  setActiveModel(modelKey: string | null): void;

  /**
   * Update storage usage display
   */
  setStorageUsage(used: number, total: number): void;

  /**
   * Get currently active model
   */
  getActiveModel(): ModelInfo | null;

  /**
   * Get all ready models (have server or browser source)
   */
  getReadyModels(): ModelInfo[];
}

export default ModelSelector;
