import type { TargetPlanMemoryLayout } from '../../config/target-plan.js';

export interface BoundSlot {
  slotId: string;
  role: string;
  scope: 'static' | 'layer-recycled' | 'transient' | 'session';
  boundAt: number;
  dimensions: Record<string, number>;
}

export interface ResourceBinder {
  bindSlots(memoryLayout: TargetPlanMemoryLayout, dynamicDimensions?: Record<string, number>): Map<string, BoundSlot>;
  getSlot(slotId: string): BoundSlot | undefined;
  releaseTransient(): void;
  releaseAll(): void;
}

export declare function createResourceBinder(device: unknown): ResourceBinder;
