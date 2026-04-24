import type {
  TensorLocation,
  TensorPhysicalStorageDescriptor,
} from './types.js';

export interface TensorPhysicalSpan {
  shardIndex: number;
  byteStart: number;
  byteEnd: number;
}

export interface TensorPackingByteLayout {
  blockElementCount: number;
  blockBytes: number | null;
}

export declare function normalizeTensorStorageDescriptor(
  storage: unknown,
  label: string
): TensorPhysicalStorageDescriptor | null;

export declare function validateTensorStorageDescriptor(
  storage: unknown,
  label: string,
  errors: string[]
): void;

export declare function getTensorPhysicalSpans(
  location: TensorLocation,
  label?: string
): TensorPhysicalSpan[];

export declare function getPackingByteLayout(
  storage: unknown,
  tensor?: { name?: string | null } | null
): TensorPackingByteLayout;
