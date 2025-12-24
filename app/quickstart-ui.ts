/**
 * quickstart-ui.ts - Quick-Start UI Component
 *
 * Provides UI panels for the quick-start download flow:
 * - Storage consent dialog with size info
 * - VRAM blocker (cannot proceed)
 * - Download progress with speed/ETA
 * - Ready state transition
 *
 * @module app/quickstart-ui
 */

import { formatBytes } from '../storage/quota.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Quick-start UI callbacks
 */
export interface QuickStartCallbacks {
  /** Called when download starts */
  onDownloadStart?: () => void;
  /** Called when download completes successfully */
  onDownloadComplete?: (modelId: string) => void;
  /** Called on download error */
  onDownloadError?: (error: Error) => void;
  /** Called when user clicks "Start Chat" */
  onRunModel?: (modelId: string) => void;
  /** Called when user cancels */
  onCancel?: () => void;
}

/**
 * Panel visibility state
 */
type PanelType = 'consent' | 'vram-blocker' | 'progress' | 'ready' | 'none';

// ============================================================================
// QuickStartUI Class
// ============================================================================

export class QuickStartUI {
  private container: HTMLElement;
  private callbacks: QuickStartCallbacks;

  // Panel elements
  private overlay: HTMLElement | null = null;
  private consentPanel: HTMLElement | null = null;
  private vramBlockerPanel: HTMLElement | null = null;
  private progressPanel: HTMLElement | null = null;
  private readyPanel: HTMLElement | null = null;

  // Consent panel elements
  private downloadSizeEl: HTMLElement | null = null;
  private storageAvailableEl: HTMLElement | null = null;
  private consentConfirmBtn: HTMLElement | null = null;
  private consentCancelBtn: HTMLElement | null = null;

  // VRAM blocker elements
  private vramRequiredEl: HTMLElement | null = null;
  private vramAvailableEl: HTMLElement | null = null;
  private vramCloseBtn: HTMLElement | null = null;

  // Progress elements
  private progressBar: HTMLElement | null = null;
  private progressPercent: HTMLElement | null = null;
  private progressSpeed: HTMLElement | null = null;
  private progressEta: HTMLElement | null = null;
  private progressDetail: HTMLElement | null = null;

  // Ready panel elements
  private readyRunBtn: HTMLElement | null = null;

  // State
  private currentPanel: PanelType = 'none';
  private pendingModelId: string | null = null;
  private consentResolver: ((value: boolean) => void) | null = null;
  private downloadStartTime: number = 0;

  /**
   * @param container - Container element (usually document.body or #chat-container)
   * @param callbacks - Event callbacks
   */
  constructor(container: HTMLElement, callbacks: QuickStartCallbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;
    this._initElements();
    this._bindEvents();
  }

  /**
   * Initialize element references
   */
  private _initElements(): void {
    this.overlay = this.container.querySelector('#quickstart-overlay');
    if (!this.overlay) return;

    // Panels
    this.consentPanel = this.overlay.querySelector('#quickstart-consent');
    this.vramBlockerPanel = this.overlay.querySelector('#quickstart-vram-blocker');
    this.progressPanel = this.overlay.querySelector('#quickstart-progress');
    this.readyPanel = this.overlay.querySelector('#quickstart-ready');

    // Consent elements
    this.downloadSizeEl = this.overlay.querySelector('#quickstart-download-size');
    this.storageAvailableEl = this.overlay.querySelector('#quickstart-storage-available');
    this.consentConfirmBtn = this.overlay.querySelector('#quickstart-confirm');
    this.consentCancelBtn = this.overlay.querySelector('#quickstart-cancel');

    // VRAM blocker elements
    this.vramRequiredEl = this.overlay.querySelector('#quickstart-vram-required');
    this.vramAvailableEl = this.overlay.querySelector('#quickstart-vram-available');
    this.vramCloseBtn = this.overlay.querySelector('#quickstart-blocker-close');

    // Progress elements
    this.progressBar = this.overlay.querySelector('#quickstart-progress-bar');
    this.progressPercent = this.overlay.querySelector('#quickstart-progress-percent');
    this.progressSpeed = this.overlay.querySelector('#quickstart-progress-speed');
    this.progressEta = this.overlay.querySelector('#quickstart-progress-eta');
    this.progressDetail = this.overlay.querySelector('#quickstart-progress-detail');

    // Ready elements
    this.readyRunBtn = this.overlay.querySelector('#quickstart-run');
  }

  /**
   * Bind event listeners
   */
  private _bindEvents(): void {
    // Consent buttons
    this.consentConfirmBtn?.addEventListener('click', () => {
      this.consentResolver?.(true);
      this.consentResolver = null;
    });

    this.consentCancelBtn?.addEventListener('click', () => {
      this.consentResolver?.(false);
      this.consentResolver = null;
      this.hide();
      this.callbacks.onCancel?.();
    });

    // VRAM blocker close
    this.vramCloseBtn?.addEventListener('click', () => {
      this.hide();
      this.callbacks.onCancel?.();
    });

    // Ready run button
    this.readyRunBtn?.addEventListener('click', () => {
      if (this.pendingModelId) {
        this.callbacks.onRunModel?.(this.pendingModelId);
      }
      this.hide();
    });
  }

  /**
   * Show a specific panel, hide others
   */
  private _showPanel(panel: PanelType): void {
    if (!this.overlay) return;

    // Hide all panels
    this.consentPanel?.setAttribute('hidden', '');
    this.vramBlockerPanel?.setAttribute('hidden', '');
    this.progressPanel?.setAttribute('hidden', '');
    this.readyPanel?.setAttribute('hidden', '');

    // Show overlay
    this.overlay.removeAttribute('hidden');

    // Show requested panel
    switch (panel) {
      case 'consent':
        this.consentPanel?.removeAttribute('hidden');
        break;
      case 'vram-blocker':
        this.vramBlockerPanel?.removeAttribute('hidden');
        break;
      case 'progress':
        this.progressPanel?.removeAttribute('hidden');
        break;
      case 'ready':
        this.readyPanel?.removeAttribute('hidden');
        break;
      case 'none':
        this.overlay.setAttribute('hidden', '');
        break;
    }

    this.currentPanel = panel;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Show storage consent dialog
   *
   * @param modelName - Display name of the model
   * @param downloadSize - Download size in bytes
   * @param availableSpace - Available storage in bytes
   * @returns Promise that resolves to true if user consents, false if declines
   */
  showStorageConsent(
    modelName: string,
    downloadSize: number,
    availableSpace: number
  ): Promise<boolean> {
    // Update text
    if (this.downloadSizeEl) {
      this.downloadSizeEl.textContent = formatBytes(downloadSize);
    }
    if (this.storageAvailableEl) {
      this.storageAvailableEl.textContent = formatBytes(availableSpace);
    }

    // Show panel
    this._showPanel('consent');

    // Return promise that resolves on button click
    return new Promise((resolve) => {
      this.consentResolver = resolve;
    });
  }

  /**
   * Show VRAM blocker (cannot proceed)
   *
   * @param requiredBytes - Required VRAM in bytes
   * @param availableBytes - Available VRAM in bytes
   */
  showVRAMBlocker(requiredBytes: number, availableBytes: number): void {
    if (this.vramRequiredEl) {
      this.vramRequiredEl.textContent = formatBytes(requiredBytes);
    }
    if (this.vramAvailableEl) {
      this.vramAvailableEl.textContent = formatBytes(availableBytes);
    }

    this._showPanel('vram-blocker');
  }

  /**
   * Show download progress panel
   */
  showDownloadProgress(): void {
    this.downloadStartTime = Date.now();
    this._showPanel('progress');
    this.setDownloadProgress(0, 0, 0, 0);
    this.callbacks.onDownloadStart?.();
  }

  /**
   * Update download progress
   *
   * @param percent - Progress 0-100
   * @param downloadedBytes - Bytes downloaded
   * @param totalBytes - Total bytes
   * @param speed - Speed in bytes/sec
   */
  setDownloadProgress(
    percent: number,
    downloadedBytes: number,
    totalBytes: number,
    speed: number
  ): void {
    if (this.progressBar) {
      this.progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
    }

    if (this.progressPercent) {
      this.progressPercent.textContent = `${Math.round(percent)}%`;
    }

    if (this.progressSpeed) {
      const speedMBs = speed / (1024 * 1024);
      this.progressSpeed.textContent = speed > 0 ? `${speedMBs.toFixed(1)} MB/s` : '-- MB/s';
    }

    if (this.progressEta && speed > 0 && totalBytes > downloadedBytes) {
      const remainingBytes = totalBytes - downloadedBytes;
      const remainingSeconds = remainingBytes / speed;

      if (remainingSeconds < 60) {
        this.progressEta.textContent = `${Math.round(remainingSeconds)}s remaining`;
      } else if (remainingSeconds < 3600) {
        const minutes = Math.round(remainingSeconds / 60);
        this.progressEta.textContent = `${minutes}m remaining`;
      } else {
        this.progressEta.textContent = 'Calculating...';
      }
    } else if (this.progressEta) {
      this.progressEta.textContent = percent >= 100 ? 'Complete!' : 'Calculating...';
    }

    if (this.progressDetail) {
      this.progressDetail.textContent = `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)}`;
    }
  }

  /**
   * Show ready state
   *
   * @param modelId - Model ID that is ready
   */
  showReady(modelId: string): void {
    this.pendingModelId = modelId;
    this._showPanel('ready');
    this.callbacks.onDownloadComplete?.(modelId);
  }

  /**
   * Show error state (reuses VRAM blocker panel styling)
   *
   * @param message - Error message
   */
  showError(message: string): void {
    // Repurpose VRAM blocker for errors
    if (this.vramRequiredEl) {
      this.vramRequiredEl.textContent = 'Error';
    }
    if (this.vramAvailableEl) {
      this.vramAvailableEl.textContent = message;
    }

    this._showPanel('vram-blocker');
    this.callbacks.onDownloadError?.(new Error(message));
  }

  /**
   * Hide all overlays
   */
  hide(): void {
    this._showPanel('none');
    this.pendingModelId = null;
    this.consentResolver = null;
  }

  /**
   * Check if quick-start UI is currently visible
   */
  isVisible(): boolean {
    return this.currentPanel !== 'none';
  }

  /**
   * Get current panel type
   */
  getCurrentPanel(): PanelType {
    return this.currentPanel;
  }
}

export default QuickStartUI;
