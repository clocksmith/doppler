


export function setStatus(dotEl, textEl, state, text) {
  if (dotEl) {
    dotEl.className = `status-dot ${state}`;
  }
  if (textEl) {
    textEl.textContent = text;
  }
}

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

export function resetStats(elements) {
  if (elements.tps) elements.tps.textContent = '--';
  if (elements.memory) elements.memory.textContent = '--';
  if (elements.gpu) elements.gpu.textContent = '--';
  if (elements.kv) elements.kv.textContent = '--';
}

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

export function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function clampInputValue(input, min, max) {
  const n = parseFloat(input.value);
  if (!Number.isFinite(n)) return;
  input.value = Math.min(max, Math.max(min, n)).toString();
}




