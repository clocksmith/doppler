/**
 * progress-ui.d.ts - Multi-Phase Progress Indicator Component Type Declarations
 * Agent-D | Phase 2 | app/
 *
 * Displays stacked loading bars for different phases:
 * - Network: Downloading model from internet (only if not cached)
 * - Cache: Reading from OPFS browser storage
 * - VRAM: Uploading weights to GPU memory
 *
 * @module app/progress-ui
 */

/**
 * Progress phase type
 */
export type ProgressPhase = 'source' | 'gpu';

/**
 * Source type for progress display
 */
export type SourceType = 'network' | 'disk' | 'cache';

/**
 * Phase progress information
 */
export interface PhaseProgress {
  phase: ProgressPhase;
  percent: number;
  bytesLoaded?: number;
  totalBytes?: number;
  speed?: number;
  message?: string;
}

/**
 * Progress UI class for displaying multi-phase loading progress
 */
export declare class ProgressUI {
  /**
   * @param container - Container element for progress overlay
   */
  constructor(container: HTMLElement);

  /**
   * Set the source type (network, disk, or cache)
   * Updates the source phase label and row class (rd.css compliant)
   */
  setSourceType(type: SourceType): void;

  /**
   * Show progress overlay
   * @param title - Title text (e.g., "Loading Model")
   */
  show(title?: string): void;

  /**
   * Update a specific phase's progress
   */
  setPhaseProgress(progress: PhaseProgress): void;

  /**
   * Legacy single-bar progress (for backwards compatibility)
   * Maps to GPU phase
   */
  setProgress(percent: number, detail?: string): void;

  /**
   * Hide progress overlay
   */
  hide(): void;

  /**
   * Show indeterminate progress for a phase
   */
  showIndeterminate(phase: ProgressPhase, message?: string): void;

  /**
   * Reset phase to determinate mode
   */
  setDeterminate(phase: ProgressPhase): void;

  /**
   * Check if progress is currently visible
   */
  isShowing(): boolean;
}

export default ProgressUI;
