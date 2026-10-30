import type { FP8Encoding, FP8Policy } from '../../config/schema/fp8.schema.js';
export interface FP8StorageDescriptor {
  packing: 'fp8-block-v1';
  encoding: FP8Encoding;
  blockShape: [number, number];
  blockBytes: number;
  scaleOffsetBytes: 0;
  dataOffsetBytes: 4;
  scaleSemantics: 'multiply';
}
export function createFP8StorageDescriptor(policy: FP8Policy): FP8StorageDescriptor;
export function fp8StorageBytes(shape: readonly number[], storage: FP8StorageDescriptor): number;
export function validateFP8StorageDescriptor(input: unknown): FP8StorageDescriptor;
export function validateFP8StorageBytes(bytes: Uint8Array, shape: readonly number[], storage: unknown): FP8StorageDescriptor;

/** Admission check before GPU upload; all codes remain legal in the artifact codec. */
export function validateFP8GPUStorageBytes(bytes: Uint8Array, shape: readonly number[], storage: unknown): FP8StorageDescriptor;
