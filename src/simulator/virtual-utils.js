
import { GB } from '../config/schema/index.js';

export const MODULE = 'VirtualDevice';

export const DEFAULT_VRAM_BUDGET_BYTES = 2 * GB;

let bufferIdCounter = 0;

export function generateBufferId() {
  return `vbuf_${Date.now()}_${bufferIdCounter++}`;
}
