/**
 * quickstart-ui.d.ts - Quick-Start UI Component Type Declarations
 *
 * Provides UI panels for the quick-start download flow:
 * - Storage consent dialog with size info
 * - VRAM blocker (cannot proceed)
 * - Download progress with speed/ETA
 * - Ready state transition
 *
 * @module app/quickstart-ui
 */

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
export type PanelType = 'consent' | 'vram-blocker' | 'progress' | 'ready' | 'none';

/**
 * Quick-start UI class for managing download flow panels
 */
export declare class QuickStartUI {
  /**
   * @param container - Container element (usually document.body or #chat-container)
   * @param callbacks - Event callbacks
   */
  constructor(container: HTMLElement, callbacks?: QuickStartCallbacks);

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
  ): Promise<boolean>;

  /**
   * Show VRAM blocker (cannot proceed)
   *
   * @param requiredBytes - Required VRAM in bytes
   * @param availableBytes - Available VRAM in bytes
   */
  showVRAMBlocker(requiredBytes: number, availableBytes: number): void;

  /**
   * Show download progress panel
   */
  showDownloadProgress(): void;

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
  ): void;

  /**
   * Show ready state
   *
   * @param modelId - Model ID that is ready
   */
  showReady(modelId: string): void;

  /**
   * Show error state (reuses VRAM blocker panel styling)
   *
   * @param message - Error message
   */
  showError(message: string): void;

  /**
   * Hide all overlays
   */
  hide(): void;

  /**
   * Check if quick-start UI is currently visible
   */
  isVisible(): boolean;

  /**
   * Get current panel type
   */
  getCurrentPanel(): PanelType;
}

export default QuickStartUI;
