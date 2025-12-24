/**
 * progress-ui.ts - Multi-Phase Progress Indicator Component
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
// Types
// ============================================================================

export type ProgressPhase = 'network' | 'cache' | 'vram';

export interface PhaseProgress {
  phase: ProgressPhase;
  percent: number;
  bytesLoaded?: number;
  totalBytes?: number;
  speed?: number;
  message?: string;
}

// ============================================================================
// ProgressUI Class
// ============================================================================

export class ProgressUI {
  private container: HTMLElement;
  private overlay: HTMLElement;
  private title: HTMLElement;
  private phasesContainer: HTMLElement;
  private phases: Map<ProgressPhase, {
    row: HTMLElement;
    bar: HTMLElement;
    label: HTMLElement;
    value: HTMLElement;
  }> = new Map();
  private isVisible = false;

  // Phase configuration
  private static readonly PHASE_CONFIG: Record<ProgressPhase, { label: string; color: string }> = {
    network: { label: 'Network', color: '#3b82f6' },  // Blue - downloading from internet
    cache: { label: 'Cache', color: '#22c55e' },      // Green - reading from OPFS
    vram: { label: 'VRAM', color: '#f59e0b' },        // Amber - uploading to GPU
  };

  /**
   * @param container - Container element for progress overlay
   */
  constructor(container: HTMLElement) {
    this.container = container;
    this.overlay = container.querySelector('#progress-overlay') as HTMLElement;
    this.title = container.querySelector('#progress-title') as HTMLElement;
    this.phasesContainer = container.querySelector('#progress-phases') as HTMLElement;

    // Create phase bars if they don't exist (backwards compatibility)
    if (!this.phasesContainer) {
      this._createPhaseElements();
    } else {
      this._initPhaseElements();
    }
  }

  /**
   * Create phase elements dynamically (for backwards compatibility)
   */
  private _createPhaseElements(): void {
    const content = this.overlay.querySelector('.progress-content');
    if (!content) return;

    // Create phases container
    this.phasesContainer = document.createElement('div');
    this.phasesContainer.id = 'progress-phases';
    this.phasesContainer.className = 'progress-phases';
    content.appendChild(this.phasesContainer);

    // Create phase bars
    for (const phase of ['network', 'cache', 'vram'] as ProgressPhase[]) {
      this._createPhaseBar(phase);
    }
  }

  /**
   * Create a single phase bar
   */
  private _createPhaseBar(phase: ProgressPhase): void {
    const config = ProgressUI.PHASE_CONFIG[phase];

    const row = document.createElement('div');
    row.className = 'progress-phase-row';
    row.dataset.phase = phase;

    const label = document.createElement('span');
    label.className = 'progress-phase-label';
    label.textContent = config.label;

    const barContainer = document.createElement('div');
    barContainer.className = 'progress-bar-container';

    const bar = document.createElement('div');
    bar.className = 'progress-bar progress-phase-bar';
    bar.style.backgroundColor = config.color;
    bar.style.width = '0%';
    barContainer.appendChild(bar);

    const value = document.createElement('span');
    value.className = 'progress-phase-value';
    value.textContent = '--';

    row.appendChild(label);
    row.appendChild(barContainer);
    row.appendChild(value);
    this.phasesContainer.appendChild(row);

    this.phases.set(phase, { row, bar, label, value });
  }

  /**
   * Initialize existing phase elements from HTML
   */
  private _initPhaseElements(): void {
    for (const phase of ['network', 'cache', 'vram'] as ProgressPhase[]) {
      const row = this.phasesContainer.querySelector(`[data-phase="${phase}"]`) as HTMLElement;
      if (row) {
        this.phases.set(phase, {
          row,
          bar: row.querySelector('.progress-phase-bar') as HTMLElement,
          label: row.querySelector('.progress-phase-label') as HTMLElement,
          value: row.querySelector('.progress-phase-value') as HTMLElement,
        });
      }
    }
  }

  /**
   * Show progress overlay
   * @param title - Title text (e.g., "Loading Model")
   */
  show(title: string = 'Loading...'): void {
    if (this.title) {
      this.title.textContent = title;
    }

    // Reset all phases
    for (const [, elements] of this.phases) {
      elements.bar.style.width = '0%';
      elements.value.textContent = '--';
      elements.row.classList.remove('active', 'complete');
    }

    this.overlay.hidden = false;
    this.isVisible = true;
  }

  /**
   * Update a specific phase's progress
   */
  setPhaseProgress(progress: PhaseProgress): void {
    const elements = this.phases.get(progress.phase);
    if (!elements) return;

    const percent = Math.min(100, Math.max(0, progress.percent));
    elements.bar.style.width = `${percent}%`;
    elements.row.classList.add('active');

    // Format the value text
    let valueText: string;
    if (progress.bytesLoaded !== undefined && progress.totalBytes !== undefined) {
      const loaded = this._formatBytes(progress.bytesLoaded);
      const total = this._formatBytes(progress.totalBytes);
      if (progress.speed !== undefined && progress.speed > 0) {
        const speed = this._formatBytes(progress.speed);
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
   * Maps to VRAM phase
   */
  setProgress(percent: number, detail?: string): void {
    this.setPhaseProgress({
      phase: 'vram',
      percent,
      message: detail,
    });
  }

  /**
   * Hide progress overlay
   */
  hide(): void {
    this.overlay.hidden = true;
    this.isVisible = false;
  }

  /**
   * Show indeterminate progress for a phase
   */
  showIndeterminate(phase: ProgressPhase, message?: string): void {
    const elements = this.phases.get(phase);
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
   */
  setDeterminate(phase: ProgressPhase): void {
    const elements = this.phases.get(phase);
    if (!elements) return;
    elements.bar.style.animation = 'none';
  }

  /**
   * Check if progress is currently visible
   */
  isShowing(): boolean {
    return this.isVisible;
  }

  /**
   * Format bytes to human-readable string
   */
  private _formatBytes(bytes: number): string {
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
