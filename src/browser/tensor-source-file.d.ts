/**
 * tensor-source-file.ts - File-backed tensor source
 *
 * Wraps File/Blob inputs with a readRange interface.
 *
 * @module browser/tensor-source-file
 */

export interface TensorSource {
  name: string;
  size: number;
  sourceType?: string;
  file?: File;
  url?: string;
  readRange(offset: number, length: number): Promise<ArrayBuffer>;
  readAll(): Promise<ArrayBuffer>;
  close(): Promise<void>;
  getAuxFiles(): Promise<Record<string, unknown>>;
  cleanup?: () => Promise<void> | void;
}

export declare function isTensorSource(value: unknown): value is TensorSource;

export declare function createFileTensorSource(file: File | Blob): TensorSource;

export declare function normalizeTensorSource(input: File | Blob | TensorSource): TensorSource;
