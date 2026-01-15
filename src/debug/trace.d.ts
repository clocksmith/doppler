import type { TraceCategory } from './config.js';

export interface TraceInterface {
  loader(message: string, data?: unknown): void;
  kernels(message: string, data?: unknown): void;
  logits(message: string, data?: unknown): void;
  embed(message: string, data?: unknown): void;
  attn(layerIdx: number, message: string, data?: unknown): void;
  ffn(layerIdx: number, message: string, data?: unknown): void;
  kv(layerIdx: number, message: string, data?: unknown): void;
  sample(message: string, data?: unknown): void;
  buffers(message: string, data?: unknown): void;
  perf(message: string, data?: unknown): void;
}

export const trace: TraceInterface;
