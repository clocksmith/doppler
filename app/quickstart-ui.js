

import { formatBytes } from '../src/storage/quota.js';

// ============================================================================
// QuickStartUI Class
// ============================================================================

export class QuickStartUI {
  
  #container;
  
  #callbacks;

  // Panel elements
  
  #overlay = null;
  
  #consentPanel = null;
  
  #vramBlockerPanel = null;
  
  #progressPanel = null;
  
  #readyPanel = null;

  // Consent panel elements
  
  #downloadSizeEl = null;
  
  #storageAvailableEl = null;
  
  #consentConfirmBtn = null;
  
  #consentCancelBtn = null;

  // VRAM blocker elements
  
  #vramRequiredEl = null;
  
  #vramAvailableEl = null;
  
  #vramCloseBtn = null;

  // Progress elements
  
  #progressBar = null;
  
  #progressPercent = null;
  
  #progressSpeed = null;
  
  #progressEta = null;
  
  #progressDetail = null;

  // Ready panel elements
  
  #readyRunBtn = null;

  // State
  
  #currentPanel = 'none';
  
  #pendingModelId = null;
  
  #consentResolver = null;
  
  #downloadStartTime = 0;

  
  constructor(container, callbacks = {}) {
    this.#container = container;
    this.#callbacks = callbacks;
    this.#initElements();
    this.#bindEvents();
  }

  
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

  
  showVRAMBlocker(requiredBytes, availableBytes) {
    if (this.#vramRequiredEl) {
      this.#vramRequiredEl.textContent = formatBytes(requiredBytes);
    }
    if (this.#vramAvailableEl) {
      this.#vramAvailableEl.textContent = formatBytes(availableBytes);
    }

    this.#showPanel('vram-blocker');
  }

  
  showDownloadProgress() {
    this.#downloadStartTime = Date.now();
    this.#showPanel('progress');
    this.setDownloadProgress(0, 0, 0, 0);
    this.#callbacks.onDownloadStart?.();
  }

  
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

  
  showReady(modelId) {
    this.#pendingModelId = modelId;
    this.#showPanel('ready');
    this.#callbacks.onDownloadComplete?.(modelId);
  }

  
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

  
  hide() {
    this.#showPanel('none');
    this.#pendingModelId = null;
    this.#consentResolver = null;
  }

  
  isVisible() {
    return this.#currentPanel !== 'none';
  }

  
  getCurrentPanel() {
    return this.#currentPanel;
  }
}

export default QuickStartUI;
