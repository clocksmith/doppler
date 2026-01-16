

// ============================================================================
// ProgressUI Class
// ============================================================================



export class ProgressUI {
  
  #container;
  
  #overlay;
  
  #title;
  
  #phasesContainer;
  
  #phases = new Map();
  
  #isVisible = false;

  // Phase configuration (rd.css compliant - uses --fg for all fills)
  
  static #PHASE_CONFIG = {
    source: { label: 'Source' },    // Dynamic label based on source type
    gpu: { label: 'GPU' },          // Uploading to GPU
  };

  // Source type labels (styling via row class, not color)
  
  static #SOURCE_CONFIG = {
    network: { label: 'Network', rowClass: 'progress-phase-network' },   // border-info style
    disk: { label: 'Disk', rowClass: 'progress-phase-disk' },            // border-ghost style
    cache: { label: 'Cache', rowClass: 'progress-phase-cache' },         // border-elevated style
  };

  
  #currentSourceType = 'cache';

  
  constructor(container) {
    this.#container = container;
    this.#overlay =  (container.querySelector('#progress-overlay'));
    this.#title =  (container.querySelector('#progress-title'));
    this.#phasesContainer =  (container.querySelector('#progress-phases'));

    // Create phase bars if they don't exist (backwards compatibility)
    if (!this.#phasesContainer) {
      this.#createPhaseElements();
    } else {
      this.#initPhaseElements();
    }
  }

  
  #createPhaseElements() {
    const content = this.#overlay.querySelector('.progress-content');
    if (!content) return;

    // Create phases container
    this.#phasesContainer = document.createElement('div');
    this.#phasesContainer.id = 'progress-phases';
    this.#phasesContainer.className = 'progress-phases';
    content.appendChild(this.#phasesContainer);

    // Create phase bars (simplified: source + gpu)
    for (const phase of  (['source', 'gpu'])) {
      this.#createPhaseBar(phase);
    }
  }

  
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

  
  #initPhaseElements() {
    for (const phase of  (['source', 'gpu'])) {
      const row =  (this.#phasesContainer.querySelector(`[data-phase="${phase}"]`));
      if (row) {
        this.#phases.set(phase, {
          row,
          bar:  (row.querySelector('.progress-fill')),
          label:  (row.querySelector('.progress-phase-label')),
          value:  (row.querySelector('.progress-phase-value')),
        });
      }
    }
  }

  
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

  
  setPhaseProgress(progress) {
    const elements = this.#phases.get(progress.phase);
    if (!elements) return;

    const percent = Math.min(100, Math.max(0, progress.percent));
    elements.bar.style.width = `${percent}%`;
    elements.row.classList.add('active');

    // Format the value text
    
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

  
  setProgress(percent, detail) {
    this.setPhaseProgress({
      phase: 'gpu',
      percent,
      message: detail,
    });
  }

  
  hide() {
    this.#overlay.hidden = true;
    this.#isVisible = false;
  }

  
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

  
  setDeterminate(phase) {
    const elements = this.#phases.get(phase);
    if (!elements) return;
    elements.bar.style.animation = 'none';
  }

  
  isShowing() {
    return this.#isVisible;
  }

  
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
