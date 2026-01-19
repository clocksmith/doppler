
export const MODULE = 'VirtualDevice';

export const DEFAULT_VRAM_BUDGET_BYTES = 2 * 1024 * 1024 * 1024; // 2GB

let bufferIdCounter = 0;

export function generateBufferId() {
  return `vbuf_${Date.now()}_${bufferIdCounter++}`;
}
