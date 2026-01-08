/**
 * progress-ui.js - Multi-Phase Progress Indicator Component
 * Agent-D | Phase 2 | app/
 *
 * Displays stacked loading bars for different phases:
 * - Network: Downloading model from internet (only if not cached)
 * - Cache: Reading from OPFS browser storage
 * - VRAM: Uploading weights to GPU memory
 *
 * @module app/progress-ui
 */

// ============================================================================
// ProgressUI Class
// ============================================================================

/**
 * @typedef {Object} PhaseElements
 * @property {HTMLElement} row
 * @property {HTMLElement} bar
 * @property {HTMLElement} label
 * @property {HTMLElement} value
 */

export class ProgressUI {
  /** @type {HTMLElement} */
  #container;
  /** @type {HTMLElement} */
  #overlay;
  /** @type {HTMLElement} */
  #title;
  /** @type {HTMLElement} */
  #phasesContainer;
  /** @type {Map<import('./progress-ui.js').ProgressPhase, PhaseElements>} */
  #phases = new Map();
  /** @type {boolean} */
  #isVisible = false;

  // Phase configuration (rd.css compliant - uses --fg for all fills)
  /** @type {Record<import('./progress-ui.js').ProgressPhase, { label: string }>} */
  static #PHASE_CONFIG = {
    source: { label: 'Source' },    // Dynamic label based on source type
    gpu: { label: 'GPU' },          // Uploading to GPU
  };

  // Source type labels (styling via row class, not color)
  /** @type {Record<import('./progress-ui.js').SourceType, { label: string; rowClass: string }>} */
  static #SOURCE_CONFIG = {
    network: { label: 'Network', rowClass: 'progress-phase-network' },   // border-info style
    disk: { label: 'Disk', rowClass: 'progress-phase-disk' },            // border-ghost style
    cache: { label: 'Cache', rowClass: 'progress-phase-cache' },         // border-elevated style
  };

  /** @type {import('./progress-ui.js').SourceType} */
  #currentSourceType = 'cache';

  /**
   * @param {HTMLElement} container - Container element for progress overlay
   */
  constructor(container) {
    this.#container = container;
    this.#overlay = /** @type {HTMLElement} */ (container.querySelector('#progress-overlay'));
    this.#title = /** @type {HTMLElement} */ (container.querySelector('#progress-title'));
    this.#phasesContainer = /** @type {HTMLElement} */ (container.querySelector('#progress-phases'));

    // Create phase bars if they don't exist (backwards compatibility)
    if (!this.#phasesContainer) {
      this.#createPhaseElements();
    } else {
      this.#initPhaseElements();
    }
  }

  /**
   * Create phase elements dynamically (for backwards compatibility)
   */
  #createPhaseElements() {
    const content = this.#overlay.querySelector('.progress-content');
    if (!content) return;

    // Create phases container
    this.#phasesContainer = document.createElement('div');
    this.#phasesContainer.id = 'progress-phases';
    this.#phasesContainer.className = 'progress-phases';
    content.appendChild(this.#phasesContainer);

    // Create phase bars (simplified: source + gpu)
    for (const phase of /** @type {import('./progress-ui.js').ProgressPhase[]} */ (['source', 'gpu'])) {
      this.#createPhaseBar(phase);
    }
  }

  /**
   * Create a single phase bar (uses rd.css .progress and .progress-fill classes)
   * @param {import('./progress-ui.js').ProgressPhase} phase
   */
  #createPhaseBar(phase) {
    const config = ProgressUI.#PHASE_CONFIG[phase];

    const row = document.createElement('div');
    row.className = 'progress-phase-row';
    row.dataset.phase = phase;

    const label = document.createElement('span');
    label.className = 'progress-phase-label';
    label.textContent = config.label;

    const barContainer = document.createElement('div');
    barContainer.className = 'progress';

    const bar = document.createElement('div');
    bar.className = 'progress-fill';
    bar.style.width = '0%';
    barContainer.appendChild(bar);

    const value = document.createElement('span');
    value.className = 'progress-phase-value muted';
    value.textContent = '--';

    row.appendChild(label);
    row.appendChild(barContainer);
    row.appendChild(value);
    this.#phasesContainer.appendChild(row);

    this.#phases.set(phase, { row, bar, label, value });
  }

  /**
   * Initialize existing phase elements from HTML
   */
  #initPhaseElements() {
    for (const phase of /** @type {import('./progress-ui.js').ProgressPhase[]} */ (['source', 'gpu'])) {
      const row = /** @type {HTMLElement} */ (this.#phasesContainer.querySelector(`[data-phase="${phase}"]`));
      if (row) {
        this.#phases.set(phase, {
          row,
          bar: /** @type {HTMLElement} */ (row.querySelector('.progress-fill')),
          label: /** @type {HTMLElement} */ (row.querySelector('.progress-phase-label')),
          value: /** @type {HTMLElement} */ (row.querySelector('.progress-phase-value')),
        });
      }
    }
  }

  /**
   * Set the source type (network, disk, or cache)
   * Updates the source phase label and row class (rd.css compliant)
   * @param {import('./progress-ui.js').SourceType} type
   */
  setSourceType(type) {
    this.#currentSourceType = type;
    const sourceElements = this.#phases.get('source');
    if (sourceElements) {
      const config = ProgressUI.#SOURCE_CONFIG[type];
      sourceElements.label.textContent = config.label;
      // Remove previous source type classes and add new one
      sourceElements.row.classList.remove('progress-phase-network', 'progress-phase-disk', 'progress-phase-cache');
      sourceElements.row.classList.add(config.rowClass);
    }
  }

  /**
   * Show progress overlay
   * @param {string} [title] - Title text (e.g., "Loading Model")
   */
  show(title = 'Loading...') {
    if (this.#title) {
      this.#title.textContent = title;
    }

    // Reset all phases
    for (const [, elements] of this.#phases) {
      elements.bar.style.width = '0%';
      elements.value.textContent = '--';
      elements.row.classList.remove('active', 'complete');
    }

    this.#overlay.hidden = false;
    this.#isVisible = true;
  }

  /**
   * Update a specific phase's progress
   * @param {import('./progress-ui.js').PhaseProgress} progress
   */
  setPhaseProgress(progress) {
    const elements = this.#phases.get(progress.phase);
    if (!elements) return;

    const percent = Math.min(100, Math.max(0, progress.percent));
    elements.bar.style.width = `${percent}%`;
    elements.row.classList.add('active');

    // Format the value text
    /** @type {string} */
    let valueText;
    if (progress.bytesLoaded !== undefined && progress.totalBytes !== undefined) {
      const loaded = this.#formatBytes(progress.bytesLoaded);
      const total = this.#formatBytes(progress.totalBytes);
      if (progress.speed !== undefined && progress.speed > 0) {
        const speed = this.#formatBytes(progress.speed);
        valueText = `${loaded} / ${total} @ ${speed}/s`;
      } else {
        valueText = `${loaded} / ${total}`;
      }
    } else if (progress.message) {
      valueText = progress.message;
    } else {
      valueText = `${Math.round(percent)}%`;
    }

    elements.value.textContent = valueText;

    // Mark complete when done
    if (percent >= 100) {
      elements.row.classList.remove('active');
      elements.row.classList.add('complete');
    }
  }

  /**
   * Legacy single-bar progress (for backwards compatibility)
   * Maps to GPU phase
   * @param {number} percent
   * @param {string} [detail]
   */
  setProgress(percent, detail) {
    this.setPhaseProgress({
      phase: 'gpu',
      percent,
      message: detail,
    });
  }

  /**
   * Hide progress overlay
   */
  hide() {
    this.#overlay.hidden = true;
    this.#isVisible = false;
  }

  /**
   * Show indeterminate progress for a phase
   * @param {import('./progress-ui.js').ProgressPhase} phase
   * @param {string} [message]
   */
  showIndeterminate(phase, message) {
    const elements = this.#phases.get(phase);
    if (!elements) return;

    elements.bar.style.width = '100%';
    elements.bar.style.animation = 'indeterminate 1.5s ease-in-out infinite';
    elements.row.classList.add('active');
    if (message) {
      elements.value.textContent = message;
    }
  }

  /**
   * Reset phase to determinate mode
   * @param {import('./progress-ui.js').ProgressPhase} phase
   */
  setDeterminate(phase) {
    const elements = this.#phases.get(phase);
    if (!elements) return;
    elements.bar.style.animation = 'none';
  }

  /**
   * Check if progress is currently visible
   * @returns {boolean}
   */
  isShowing() {
    return this.#isVisible;
  }

  /**
   * Format bytes to human-readable string
   * @param {number} bytes
   * @returns {string}
   */
  #formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    const value = bytes / Math.pow(k, i);
    // Show 1 decimal for MB/GB, 0 for smaller
    const decimals = i >= 2 ? 1 : 0;
    return value.toFixed(decimals) + ' ' + sizes[i];
  }
}

export default ProgressUI;
