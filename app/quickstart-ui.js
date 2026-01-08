/**
 * quickstart-ui.js - Quick-Start UI Component
 *
 * Provides UI panels for the quick-start download flow:
 * - Storage consent dialog with size info
 * - VRAM blocker (cannot proceed)
 * - Download progress with speed/ETA
 * - Ready state transition
 *
 * @module app/quickstart-ui
 */

import { formatBytes } from '../src/storage/quota.js';

// ============================================================================
// QuickStartUI Class
// ============================================================================

export class QuickStartUI {
  /** @type {HTMLElement} */
  #container;
  /** @type {import('./quickstart-ui.js').QuickStartCallbacks} */
  #callbacks;

  // Panel elements
  /** @type {HTMLElement | null} */
  #overlay = null;
  /** @type {HTMLElement | null} */
  #consentPanel = null;
  /** @type {HTMLElement | null} */
  #vramBlockerPanel = null;
  /** @type {HTMLElement | null} */
  #progressPanel = null;
  /** @type {HTMLElement | null} */
  #readyPanel = null;

  // Consent panel elements
  /** @type {HTMLElement | null} */
  #downloadSizeEl = null;
  /** @type {HTMLElement | null} */
  #storageAvailableEl = null;
  /** @type {HTMLElement | null} */
  #consentConfirmBtn = null;
  /** @type {HTMLElement | null} */
  #consentCancelBtn = null;

  // VRAM blocker elements
  /** @type {HTMLElement | null} */
  #vramRequiredEl = null;
  /** @type {HTMLElement | null} */
  #vramAvailableEl = null;
  /** @type {HTMLElement | null} */
  #vramCloseBtn = null;

  // Progress elements
  /** @type {HTMLElement | null} */
  #progressBar = null;
  /** @type {HTMLElement | null} */
  #progressPercent = null;
  /** @type {HTMLElement | null} */
  #progressSpeed = null;
  /** @type {HTMLElement | null} */
  #progressEta = null;
  /** @type {HTMLElement | null} */
  #progressDetail = null;

  // Ready panel elements
  /** @type {HTMLElement | null} */
  #readyRunBtn = null;

  // State
  /** @type {import('./quickstart-ui.js').PanelType} */
  #currentPanel = 'none';
  /** @type {string | null} */
  #pendingModelId = null;
  /** @type {((value: boolean) => void) | null} */
  #consentResolver = null;
  /** @type {number} */
  #downloadStartTime = 0;

  /**
   * @param {HTMLElement} container - Container element (usually document.body or #chat-container)
   * @param {import('./quickstart-ui.js').QuickStartCallbacks} [callbacks] - Event callbacks
   */
  constructor(container, callbacks = {}) {
    this.#container = container;
    this.#callbacks = callbacks;
    this.#initElements();
    this.#bindEvents();
  }

  /**
   * Initialize element references
   */
  #initElements() {
    this.#overlay = this.#container.querySelector('#quickstart-overlay');
    if (!this.#overlay) return;

    // Panels
    this.#consentPanel = this.#overlay.querySelector('#quickstart-consent');
    this.#vramBlockerPanel = this.#overlay.querySelector('#quickstart-vram-blocker');
    this.#progressPanel = this.#overlay.querySelector('#quickstart-progress');
    this.#readyPanel = this.#overlay.querySelector('#quickstart-ready');

    // Consent elements
    this.#downloadSizeEl = this.#overlay.querySelector('#quickstart-download-size');
    this.#storageAvailableEl = this.#overlay.querySelector('#quickstart-storage-available');
    this.#consentConfirmBtn = this.#overlay.querySelector('#quickstart-confirm');
    this.#consentCancelBtn = this.#overlay.querySelector('#quickstart-cancel');

    // VRAM blocker elements
    this.#vramRequiredEl = this.#overlay.querySelector('#quickstart-vram-required');
    this.#vramAvailableEl = this.#overlay.querySelector('#quickstart-vram-available');
    this.#vramCloseBtn = this.#overlay.querySelector('#quickstart-blocker-close');

    // Progress elements
    this.#progressBar = this.#overlay.querySelector('#quickstart-progress-bar');
    this.#progressPercent = this.#overlay.querySelector('#quickstart-progress-percent');
    this.#progressSpeed = this.#overlay.querySelector('#quickstart-progress-speed');
    this.#progressEta = this.#overlay.querySelector('#quickstart-progress-eta');
    this.#progressDetail = this.#overlay.querySelector('#quickstart-progress-detail');

    // Ready elements
    this.#readyRunBtn = this.#overlay.querySelector('#quickstart-run');
  }

  /**
   * Bind event listeners
   */
  #bindEvents() {
    // Consent buttons
    this.#consentConfirmBtn?.addEventListener('click', () => {
      this.#consentResolver?.(true);
      this.#consentResolver = null;
    });

    this.#consentCancelBtn?.addEventListener('click', () => {
      this.#consentResolver?.(false);
      this.#consentResolver = null;
      this.hide();
      this.#callbacks.onCancel?.();
    });

    // VRAM blocker close
    this.#vramCloseBtn?.addEventListener('click', () => {
      this.hide();
      this.#callbacks.onCancel?.();
    });

    // Ready run button
    this.#readyRunBtn?.addEventListener('click', () => {
      if (this.#pendingModelId) {
        this.#callbacks.onRunModel?.(this.#pendingModelId);
      }
      this.hide();
    });
  }

  /**
   * Show a specific panel, hide others
   * @param {import('./quickstart-ui.js').PanelType} panel
   */
  #showPanel(panel) {
    if (!this.#overlay) return;

    // Hide all panels
    this.#consentPanel?.setAttribute('hidden', '');
    this.#vramBlockerPanel?.setAttribute('hidden', '');
    this.#progressPanel?.setAttribute('hidden', '');
    this.#readyPanel?.setAttribute('hidden', '');

    // Show overlay
    this.#overlay.removeAttribute('hidden');

    // Show requested panel
    switch (panel) {
      case 'consent':
        this.#consentPanel?.removeAttribute('hidden');
        break;
      case 'vram-blocker':
        this.#vramBlockerPanel?.removeAttribute('hidden');
        break;
      case 'progress':
        this.#progressPanel?.removeAttribute('hidden');
        break;
      case 'ready':
        this.#readyPanel?.removeAttribute('hidden');
        break;
      case 'none':
        this.#overlay.setAttribute('hidden', '');
        break;
    }

    this.#currentPanel = panel;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Show storage consent dialog
   *
   * @param {string} modelName - Display name of the model
   * @param {number} downloadSize - Download size in bytes
   * @param {number} availableSpace - Available storage in bytes
   * @returns {Promise<boolean>} Promise that resolves to true if user consents, false if declines
   */
  showStorageConsent(
    modelName,
    downloadSize,
    availableSpace
  ) {
    // Update text
    if (this.#downloadSizeEl) {
      this.#downloadSizeEl.textContent = formatBytes(downloadSize);
    }
    if (this.#storageAvailableEl) {
      this.#storageAvailableEl.textContent = formatBytes(availableSpace);
    }

    // Show panel
    this.#showPanel('consent');

    // Return promise that resolves on button click
    return new Promise((resolve) => {
      this.#consentResolver = resolve;
    });
  }

  /**
   * Show VRAM blocker (cannot proceed)
   *
   * @param {number} requiredBytes - Required VRAM in bytes
   * @param {number} availableBytes - Available VRAM in bytes
   */
  showVRAMBlocker(requiredBytes, availableBytes) {
    if (this.#vramRequiredEl) {
      this.#vramRequiredEl.textContent = formatBytes(requiredBytes);
    }
    if (this.#vramAvailableEl) {
      this.#vramAvailableEl.textContent = formatBytes(availableBytes);
    }

    this.#showPanel('vram-blocker');
  }

  /**
   * Show download progress panel
   */
  showDownloadProgress() {
    this.#downloadStartTime = Date.now();
    this.#showPanel('progress');
    this.setDownloadProgress(0, 0, 0, 0);
    this.#callbacks.onDownloadStart?.();
  }

  /**
   * Update download progress
   *
   * @param {number} percent - Progress 0-100
   * @param {number} downloadedBytes - Bytes downloaded
   * @param {number} totalBytes - Total bytes
   * @param {number} speed - Speed in bytes/sec
   */
  setDownloadProgress(
    percent,
    downloadedBytes,
    totalBytes,
    speed
  ) {
    if (this.#progressBar) {
      this.#progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
    }

    if (this.#progressPercent) {
      this.#progressPercent.textContent = `${Math.round(percent)}%`;
    }

    if (this.#progressSpeed) {
      const speedMBs = speed / (1024 * 1024);
      this.#progressSpeed.textContent = speed > 0 ? `${speedMBs.toFixed(1)} MB/s` : '-- MB/s';
    }

    if (this.#progressEta && speed > 0 && totalBytes > downloadedBytes) {
      const remainingBytes = totalBytes - downloadedBytes;
      const remainingSeconds = remainingBytes / speed;

      if (remainingSeconds < 60) {
        this.#progressEta.textContent = `${Math.round(remainingSeconds)}s remaining`;
      } else if (remainingSeconds < 3600) {
        const minutes = Math.round(remainingSeconds / 60);
        this.#progressEta.textContent = `${minutes}m remaining`;
      } else {
        this.#progressEta.textContent = 'Calculating...';
      }
    } else if (this.#progressEta) {
      this.#progressEta.textContent = percent >= 100 ? 'Complete!' : 'Calculating...';
    }

    if (this.#progressDetail) {
      this.#progressDetail.textContent = `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)}`;
    }
  }

  /**
   * Show ready state
   *
   * @param {string} modelId - Model ID that is ready
   */
  showReady(modelId) {
    this.#pendingModelId = modelId;
    this.#showPanel('ready');
    this.#callbacks.onDownloadComplete?.(modelId);
  }

  /**
   * Show error state (reuses VRAM blocker panel styling)
   *
   * @param {string} message - Error message
   */
  showError(message) {
    // Repurpose VRAM blocker for errors
    if (this.#vramRequiredEl) {
      this.#vramRequiredEl.textContent = 'Error';
    }
    if (this.#vramAvailableEl) {
      this.#vramAvailableEl.textContent = message;
    }

    this.#showPanel('vram-blocker');
    this.#callbacks.onDownloadError?.(new Error(message));
  }

  /**
   * Hide all overlays
   */
  hide() {
    this.#showPanel('none');
    this.#pendingModelId = null;
    this.#consentResolver = null;
  }

  /**
   * Check if quick-start UI is currently visible
   * @returns {boolean}
   */
  isVisible() {
    return this.#currentPanel !== 'none';
  }

  /**
   * Get current panel type
   * @returns {import('./quickstart-ui.js').PanelType}
   */
  getCurrentPanel() {
    return this.#currentPanel;
  }
}

export default QuickStartUI;
