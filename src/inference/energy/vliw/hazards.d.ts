export function buildHazardDeps(tasks: Array<{ id: number; reads?: number[]; writes?: number[] }>): number[][];
export function buildUnifiedDeps(tasks: Array<{ id: number; deps?: number[]; reads?: number[]; writes?: number[] }>): number[][];
