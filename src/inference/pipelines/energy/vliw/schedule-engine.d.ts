export function computeEngineOffsets(caps: Record<string, number>): {
  offsets: Record<string, number>;
  totalSlots: number;
  totalSlotsNonDebug: number;
  slotEngines: string[];
  slotIndices: number[];
};
