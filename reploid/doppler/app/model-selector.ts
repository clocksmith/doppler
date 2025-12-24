/**
 * model-selector.ts - Model Selection Component
 * Agent-D | Phase 2 | app/
 *
 * Handles model list display, download progress, and selection.
 *
 * @module app/model-selector
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

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

// ============================================================================
// ModelSelector Class
// ============================================================================

export class ModelSelector {
  private container: HTMLElement;
  private listElement: HTMLElement;
  private storageElement: HTMLElement;

  private onSelect: (model: ModelInfo, opts?: { preferredSource?: string }) => void;
  private onDownload: (model: ModelInfo, opts?: { runAfter?: boolean }) => void;
  private onDelete: (model: ModelInfo) => void;
  private onQuickStart: (model: ModelInfo) => void;

  private models: ModelInfo[] = [];
  private activeModelId: string | null = null;
  private downloadingModelId: string | null = null;

  /**
   * @param container - Container element for model list
   * @param callbacks - Event callbacks
   */
  constructor(container: HTMLElement, callbacks: ModelSelectorCallbacks = {}) {
    this.container = container;
    this.listElement = container.querySelector('#model-list') as HTMLElement;
    this.storageElement = container.querySelector('#storage-used') as HTMLElement;

    this.onSelect = callbacks.onSelect || (() => {});
    this.onDownload = callbacks.onDownload || (() => {});
    this.onDelete = callbacks.onDelete || (() => {});
    this.onQuickStart = callbacks.onQuickStart || (() => {});
  }

  /**
   * Set available models
   */
  setModels(models: ModelInfo[]): void {
    this.models = models;
    this._render();
  }

  /**
   * Update a single model's info
   */
  updateModel(modelKey: string, updates: Partial<ModelInfo>): void {
    const model = this.models.find((m) => m.key === modelKey);
    if (model) {
      Object.assign(model, updates);
      this._render();
    }
  }

  /**
   * Set download progress for a model
   */
  setDownloadProgress(modelKey: string, progress: number): void {
    this.downloadingModelId = progress < 100 ? modelKey : null;
    this.updateModel(modelKey, { downloadProgress: progress });
  }

  /**
   * Mark a model as downloaded (triggers refresh)
   */
  setDownloaded(modelKey: string): void {
    this.downloadingModelId = null;
    this.updateModel(modelKey, { downloadProgress: undefined });
  }

  /**
   * Set the active (running) model
   */
  setActiveModel(modelKey: string | null): void {
    this.activeModelId = modelKey;
    this._render();
  }

  /**
   * Update storage usage display
   */
  setStorageUsage(used: number, total: number): void {
    const usedStr = this._formatBytes(used);
    const totalStr = this._formatBytes(total);
    this.storageElement.textContent = `${usedStr} / ${totalStr}`;
  }

  /**
   * Render the model list with grouped sections
   */
  private _render(): void {
    this.listElement.innerHTML = '';

    if (this.models.length === 0) {
      this.listElement.innerHTML = `
        <div class="model-item" style="text-align: center; color: var(--text-muted);">
          No models available
        </div>
      `;
      return;
    }

    // Group models: "Ready" (has server or browser) vs "Available" (remote only)
    const readyModels = this.models.filter(
      (m) => m.sources?.server || m.sources?.browser
    );
    const availableModels = this.models.filter(
      (m) => !m.sources?.server && !m.sources?.browser && m.sources?.remote
    );

    // Render ready models section
    if (readyModels.length > 0) {
      const section = this._createSection(
        'Ready',
        `${readyModels.length} model${readyModels.length > 1 ? 's' : ''}`,
        'ready'
      );
      for (const model of readyModels) {
        section.appendChild(this._createModelItem(model));
      }
      this.listElement.appendChild(section);
    }

    // Render available for download section
    if (availableModels.length > 0) {
      const section = this._createSection('Available', 'Download to browser', 'remote');
      for (const model of availableModels) {
        section.appendChild(this._createModelItem(model));
      }
      this.listElement.appendChild(section);
    }
  }

  /**
   * Create a section header element
   */
  private _createSection(title: string, subtitle: string, type: string): HTMLElement {
    const section = document.createElement('div');
    section.className = `model-section model-section-${type}`;
    section.innerHTML = `
      <div class="model-section-header">
        <span class="model-section-title">${title}</span>
        <span class="model-section-subtitle">${subtitle}</span>
      </div>
    `;
    return section;
  }

  /**
   * Create a model list item element
   */
  private _createModelItem(model: ModelInfo): HTMLElement {
    const item = document.createElement('div');
    item.className = 'model-item';
    item.dataset.modelKey = model.key;

    const sources = model.sources || {};
    const hasServer = !!sources.server;
    const hasBrowser = !!sources.browser;
    const hasRemote = !!sources.remote;
    const isReady = hasServer || hasBrowser;

    if (model.key === this.activeModelId) {
      item.classList.add('active');
    }

    if (model.key === this.downloadingModelId) {
      item.classList.add('downloading');
      item.style.setProperty('--download-progress', `${model.downloadProgress || 0}%`);
    }

    const isDownloading = model.key === this.downloadingModelId;

    // Build meta info
    const metaParts: string[] = [];
    if (model.architecture) metaParts.push(model.architecture);
    if (model.size && model.size !== 'Unknown') metaParts.push(model.size);
    if (model.quantization && model.quantization !== 'Unknown') metaParts.push(model.quantization);
    if (model.downloadSize && model.downloadSize > 0) metaParts.push(this._formatBytes(model.downloadSize));
    const metaText = metaParts.join(' · ') || 'Unknown';

    const isLoaded = model.key === this.activeModelId;

    // Build tooltips based on source
    const serverRunTooltip = 'Load weights from dev server into GPU memory.';
    const cachedRunTooltip = 'Load weights from browser cache (OPFS) into GPU memory.';
    const cacheTooltip = `Copy model from server to browser storage (OPFS). Uses ~${this._formatBytes(model.downloadSize || 0)} of browser storage.`;
    const clearCacheTooltip = 'Remove cached copy from browser storage.';
    const deleteTooltip = 'Permanently remove model from browser storage.';
    const downloadTooltip = `Download model to browser storage (OPFS). Uses ~${this._formatBytes(model.downloadSize || 0)} of browser storage.`;
    const dlRunTooltip = 'Download to browser storage, then immediately load into GPU memory.';

    // Build action buttons based on available sources
    let actionsHtml = '';
    if (isReady) {
      if (hasServer && hasBrowser) {
        actionsHtml = `
          <button class="model-btn run run-server" data-source="server" ${isDownloading || isLoaded ? 'disabled' : ''} title="${serverRunTooltip}">
            Run (Server)
          </button>
          <button class="model-btn run run-cached" data-source="browser" ${isDownloading || isLoaded ? 'disabled' : ''} title="${cachedRunTooltip}">
            Run (Cached)
          </button>
          <button class="model-btn delete" ${isDownloading ? 'disabled' : ''} title="${clearCacheTooltip}">
            Clear Cache
          </button>
        `;
      } else if (hasServer) {
        actionsHtml = `
          <button class="model-btn run" data-source="server" ${isDownloading || isLoaded ? 'disabled' : ''} title="${serverRunTooltip}">
            ${isLoaded ? 'Running' : 'Run (Server)'}
          </button>
          <button class="model-btn cache" ${isDownloading ? 'disabled' : ''} title="${cacheTooltip}">
            Cache
          </button>
        `;
      } else {
        actionsHtml = `
          <button class="model-btn run" data-source="browser" ${isDownloading || isLoaded ? 'disabled' : ''} title="${cachedRunTooltip}">
            ${isLoaded ? 'Running' : 'Run (Cached)'}
          </button>
          <button class="model-btn delete" ${isDownloading ? 'disabled' : ''} title="${deleteTooltip}">
            Delete
          </button>
        `;
      }
    } else if (hasRemote) {
      const quickStartTooltip = 'One-click setup: checks VRAM, downloads to browser storage, then runs.';
      if (model.quickStartAvailable) {
        actionsHtml = `
          <button class="model-btn quick-start" ${isDownloading ? 'disabled' : ''} title="${quickStartTooltip}">
            Quick Start
          </button>
          <button class="model-btn download" ${isDownloading ? 'disabled' : ''} title="${downloadTooltip}">
            ${isDownloading ? `${Math.round(model.downloadProgress || 0)}%` : 'Download'}
          </button>
        `;
      } else {
        actionsHtml = `
          <button class="model-btn download" ${isDownloading ? 'disabled' : ''} title="${downloadTooltip}">
            ${isDownloading ? `${Math.round(model.downloadProgress || 0)}%` : 'Download'}
          </button>
          <button class="model-btn download-run" ${isDownloading ? 'disabled' : ''} title="${dlRunTooltip}">
            DL & Run
          </button>
        `;
      }
    }

    item.innerHTML = `
      <div class="model-name">${this._escapeHtml(model.name)}</div>
      <div class="model-meta">${metaText}</div>
      <div class="model-actions">${actionsHtml}</div>
    `;

    // Bind events
    const runBtns = item.querySelectorAll('.model-btn.run');
    const cacheBtn = item.querySelector('.model-btn.cache');
    const downloadBtn = item.querySelector('.model-btn.download');
    const downloadRunBtn = item.querySelector('.model-btn.download-run');
    const deleteBtn = item.querySelector('.model-btn.delete');
    const quickStartBtn = item.querySelector('.model-btn.quick-start');

    runBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (model.key !== this.activeModelId) {
          const preferredSource = (btn as HTMLElement).dataset.source || undefined;
          this.onSelect(model, { preferredSource });
        }
      });
    });

    if (cacheBtn) {
      cacheBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onDownload(model);
      });
    }

    if (downloadBtn) {
      downloadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onDownload(model);
      });
    }

    if (downloadRunBtn) {
      downloadRunBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onDownload(model, { runAfter: true });
      });
    }

    if (quickStartBtn) {
      quickStartBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onQuickStart(model);
      });
    }

    if (deleteBtn) {
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const msg = hasServer
          ? `Clear browser cache for ${model.name}? (Server copy will remain)`
          : `Delete ${model.name}? This will remove it from browser storage.`;
        if (confirm(msg)) {
          this.onDelete(model);
        }
      });
    }

    // Click on item to run (if ready)
    item.addEventListener('click', () => {
      if (isReady && model.key !== this.activeModelId) {
        this.onSelect(model);
      }
    });

    return item;
  }

  /**
   * Format bytes to human-readable string
   */
  private _formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  /**
   * Escape HTML to prevent XSS
   */
  private _escapeHtml(str: string): string {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  /**
   * Get currently active model
   */
  getActiveModel(): ModelInfo | null {
    return this.models.find((m) => m.key === this.activeModelId) || null;
  }

  /**
   * Get all ready models (have server or browser source)
   */
  getReadyModels(): ModelInfo[] {
    return this.models.filter((m) => m.sources?.server || m.sources?.browser);
  }
}

export default ModelSelector;
