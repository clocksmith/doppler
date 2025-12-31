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
        <div class="model-item muted" style="text-align: center;">
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

    // Determine source badge and tooltips
    const sourceUrl = sources.server?.url || sources.browser?.url || sources.remote?.url || '';
    const isLocalServer = sourceUrl.match(/^(https?:\/\/)?(localhost|127\.0\.0\.1|0\.0\.0\.0|file:)/i);

    let sourceBadge = '';
    let sourceTooltip = '';
    let runTooltip = '';
    let secondaryBtn = '';

    if (hasServer && hasBrowser) {
      // Both server and cache available - prefer cache, show toggle
      sourceBadge = '<span class="badge badge-filled" title="Cached in browser">Cache</span>';
      sourceTooltip = 'Also available from dev server';
      runTooltip = 'Load from browser cache into GPU memory';
      secondaryBtn = `<button class="btn delete" ${isDownloading ? 'disabled' : ''} title="Remove cached copy from browser storage">Clear</button>`;
    } else if (hasServer) {
      sourceBadge = isLocalServer
        ? '<span class="badge border-ghost" title="Local dev server">Disk</span>'
        : '<span class="badge border-info" title="Remote server">Network</span>';
      runTooltip = 'Load weights from server into GPU memory';
      secondaryBtn = `<button class="btn cache" ${isDownloading ? 'disabled' : ''} title="Copy to browser storage (~${this._formatBytes(model.downloadSize || 0)})">Save</button>`;
    } else if (hasBrowser) {
      sourceBadge = '<span class="badge badge-filled" title="Cached in browser">Cache</span>';
      runTooltip = 'Load from browser cache into GPU memory';
      secondaryBtn = `<button class="btn delete" ${isDownloading ? 'disabled' : ''} title="Remove from browser storage">Delete</button>`;
    } else if (hasRemote) {
      sourceBadge = '<span class="badge border-info" title="Download required">Network</span>';
      runTooltip = 'Download to browser storage, then load into GPU memory';
      secondaryBtn = `<button class="btn download-only" ${isDownloading ? 'disabled' : ''} title="Download without running (~${this._formatBytes(model.downloadSize || 0)})">Save</button>`;
    }

    // Build unified action buttons
    let actionsHtml = '';
    const preferredSource = hasBrowser ? 'browser' : (hasServer ? 'server' : 'remote');

    if (isReady || hasRemote) {
      actionsHtml = `
        <button class="btn btn-primary run" data-source="${preferredSource}" ${isDownloading || isLoaded ? 'disabled' : ''} title="${runTooltip}">
          ${isDownloading ? `${Math.round(model.downloadProgress || 0)}%` : (isLoaded ? 'Running' : 'Run')}
        </button>
        ${secondaryBtn}
      `;
    }

    item.innerHTML = `
      <div class="flex justify-between items-center">
        <div class="type-h2">${this._escapeHtml(model.name)}</div>
        ${sourceBadge}
      </div>
      <div class="type-caption muted">${metaText}</div>
      <div class="model-actions">${actionsHtml}</div>
    `;

    // Bind events
    const runBtn = item.querySelector('.btn.run');
    const cacheBtn = item.querySelector('.btn.cache');
    const downloadOnlyBtn = item.querySelector('.btn.download-only');
    const deleteBtn = item.querySelector('.btn.delete');

    if (runBtn) {
      runBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (model.key !== this.activeModelId) {
          const source = (runBtn as HTMLElement).dataset.source;
          if (source === 'remote') {
            // Remote models: trigger quick-start flow (VRAM check -> download -> run)
            this.onQuickStart(model);
          } else {
            // Local models: run directly
            this.onSelect(model, { preferredSource: source });
          }
        }
      });
    }

    if (cacheBtn) {
      cacheBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onDownload(model);
      });
    }

    if (downloadOnlyBtn) {
      downloadOnlyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.onDownload(model);
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

    // Click on item to run (if ready or remote)
    item.addEventListener('click', () => {
      if (model.key !== this.activeModelId) {
        if (isReady) {
          this.onSelect(model);
        } else if (hasRemote) {
          this.onQuickStart(model);
        }
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
