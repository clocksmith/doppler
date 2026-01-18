

/**
 * UI helper functions for status, stats, and error display.
 */

/**
 * Update the status indicator.
 * @param {HTMLElement|null} dotEl
 * @param {HTMLElement|null} textEl
 * @param {'ready'|'loading'|'error'} state
 * @param {string} text
 */
export function setStatus(dotEl, textEl, state, text) {
  if (dotEl) {
    dotEl.className = `status-dot ${state}`;
  }
  if (textEl) {
    textEl.textContent = text;
  }
}

/**
 * Update capabilities UI list.
 * @param {HTMLElement|null} listEl
 * @param {Record<string, boolean>} capabilities
 */
export function updateCapabilitiesUI(listEl, capabilities) {
  if (!listEl) return;

  const items = listEl.querySelectorAll('li');
  items.forEach((item) => {
    const cap = item.dataset.cap;
    if (cap && cap in capabilities) {
      if (capabilities[cap]) {
        item.classList.add('supported');
        item.classList.remove('unsupported');
      } else {
        item.classList.add('unsupported');
        item.classList.remove('supported');
      }
    }
  });
}

/**
 * Populate GPU info panel.
 * @param {GPUInfoElements} elements
 * @param {GPUInfoData} data
 */
export function populateGPUInfo(elements, data) {
  if (!elements.device) return;

  // Device name
  elements.device.textContent = data.deviceName;
  elements.device.title = data.deviceName;

  // VRAM/buffer limit
  if (elements.vram) {
    elements.vram.textContent = data.bufferLimitBytes > 0
      ? formatBytes(data.bufferLimitBytes)
      : 'Unknown';
  }

  // System RAM (for unified memory)
  if (data.isUnifiedMemory && elements.ramRow && elements.ram) {
    if (data.systemMemoryGB && data.systemMemoryGB >= 8) {
      elements.ram.textContent = '8+ GB';
    } else if (data.systemMemoryGB) {
      elements.ram.textContent = `${data.systemMemoryGB} GB`;
    } else {
      elements.ram.textContent = 'Unknown';
    }
    elements.ramRow.hidden = false;
  }

  // Unified memory note
  if (data.isUnifiedMemory && elements.unifiedNote) {
    elements.unifiedNote.hidden = false;
  }

  // Features
  if (elements.features) {
    const features = [];
    if (data.hasF16) features.push('F16');
    if (data.hasSubgroups) features.push('Subgroups');
    if (data.hasTimestamps) features.push('Timestamps');
    elements.features.textContent = features.length > 0 ? features.join(', ') : 'Basic';
  }
}

/**
 * Update generation stats display.
 * @param {StatsElements} elements
 * @param {StatsData} stats
 */
export function updateStats(elements, stats) {
  if (elements.tps && stats.tps !== undefined) {
    elements.tps.textContent = typeof stats.tps === 'number'
      ? stats.tps.toFixed(1)
      : stats.tps;
  }

  if (elements.memory && stats.memoryMB !== undefined) {
    elements.memory.textContent = `${stats.memoryMB} MB`;
  }

  if (elements.gpu && stats.gpuBuffers !== undefined) {
    elements.gpu.textContent = `${stats.gpuBuffers}`;
  }

  if (elements.kv && stats.kvSeqLen !== undefined && stats.kvMaxLen !== undefined) {
    elements.kv.textContent = `${stats.kvSeqLen}/${stats.kvMaxLen}`;
  }
}

/**
 * Reset stats to initial "ready" state.
 * @param {StatsElements} elements
 */
export function resetStats(elements) {
  if (elements.tps) elements.tps.textContent = '--';
  if (elements.memory) elements.memory.textContent = '--';
  if (elements.gpu) elements.gpu.textContent = '--';
  if (elements.kv) elements.kv.textContent = '--';
}

/**
 * Show an error modal.
 * @param {HTMLElement|null} modalEl
 * @param {HTMLElement|null} messageEl
 * @param {HTMLElement|null} closeBtn
 * @param {string} message
 */
export function showError(modalEl, messageEl, closeBtn, message) {
  if (messageEl) {
    messageEl.textContent = message;
  }
  if (modalEl) {
    modalEl.hidden = false;
  }

  const close = () => {
    if (modalEl) {
      modalEl.hidden = true;
    }
    closeBtn?.removeEventListener('click', close);
  };
  closeBtn?.addEventListener('click', close);
}

/**
 * Format bytes to human-readable string.
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Clamp a numeric input value.
 * @param {HTMLInputElement} input
 * @param {number} min
 * @param {number} max
 */
export function clampInputValue(input, min, max) {
  const n = parseFloat(input.value);
  if (!Number.isFinite(n)) return;
  input.value = Math.min(max, Math.max(min, n)).toString();
}

/**
 * @typedef {Object} GPUInfoElements
 * @property {HTMLElement|null} device
 * @property {HTMLElement|null} vram
 * @property {HTMLElement|null} vramLabel
 * @property {HTMLElement|null} ram
 * @property {HTMLElement|null} ramRow
 * @property {HTMLElement|null} features
 * @property {HTMLElement|null} unifiedNote
 */

/**
 * @typedef {Object} GPUInfoData
 * @property {string} deviceName
 * @property {number} bufferLimitBytes
 * @property {boolean} isUnifiedMemory
 * @property {number} [systemMemoryGB]
 * @property {boolean} hasF16
 * @property {boolean} hasSubgroups
 * @property {boolean} hasTimestamps
 */

/**
 * @typedef {Object} StatsElements
 * @property {HTMLElement|null} tps
 * @property {HTMLElement|null} memory
 * @property {HTMLElement|null} gpu
 * @property {HTMLElement|null} kv
 */

/**
 * @typedef {Object} StatsData
 * @property {number|string} [tps]
 * @property {number} [memoryMB]
 * @property {number} [gpuBuffers]
 * @property {number} [kvSeqLen]
 * @property {number} [kvMaxLen]
 */
