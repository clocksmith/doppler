import type { TargetPlanMemoryExpression, TargetPlanMemoryLayout } from '../../config/target-plan.js';

export interface BoundSlot {
  slotId: string;
  role: string;
  scope: 'static' | 'layer-recycled' | 'transient' | 'session';
  owner: 'runtime' | 'program';
  sizeBytes: number;
  dimensions: Record<string, number>;
  buffer?: GPUBuffer;
  resource?: unknown;
}

export interface ResourceBinder {
  assertDeviceAvailable(): void;
  bindSlots(memoryLayout: TargetPlanMemoryLayout, dimensions: Record<string, number>): Map<string, BoundSlot>;
  writeSlot(slotId: string, data: ArrayBuffer | ArrayBufferView, offset?: number): void;
  getSlot(slotId: string): BoundSlot | undefined;
  releaseTransient(): void;
  releaseAll(): void;
}

export declare function evaluateMemoryExpression(expression: TargetPlanMemoryExpression, dimensions: Record<string, number>): number;
export declare function createResourceBinder(device: unknown, program?: unknown): ResourceBinder;
