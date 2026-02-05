export function scheduleFromBundles(tasks: Array<{ id: number }>, caps: Record<string, number>): {
  cycles: number;
  utilization: number;
  violations: number;
  scheduled: number;
  grid: Float32Array;
  gridShape: [number, number, number];
  slotAssignments: Int32Array;
  slotEngines: string[];
  slotIndices: number[];
};
