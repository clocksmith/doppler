import type {
  ServeFailureReceiptOptions,
  ServeReceiptBase,
  ServeReceiptOptions,
} from '../doppler-serve.js';

export function generateCompletionId(): string;
export function bindLogicalRuntimeModel(runtimeModel: unknown, requestedModel: string): unknown;
export function buildServeReceipt(options: ServeReceiptOptions): ServeReceiptBase & {
  status: 'pass';
  output: Record<string, unknown>;
  transcript: Record<string, unknown>;
  usage: ServeReceiptOptions['usage'];
};
export function buildServeFailureReceipt(options: ServeFailureReceiptOptions): ServeReceiptBase & {
  status: 'diagnostic';
  failure: Record<string, unknown>;
};
