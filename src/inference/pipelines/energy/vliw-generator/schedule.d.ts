export function scheduleOpsDepOnce(
  ops: Array<{ engine: string }>,
  caps: Record<string, number>,
  options?: { returnOps?: boolean; seed?: number; jitter?: number },
): Array<Record<string, unknown[]>>;
export function scheduleOpsDep(
  ops: Array<{ engine: string }>,
  caps: Record<string, number>,
  options?: { returnOps?: boolean; seed?: number; jitter?: number; restarts?: number },
): Array<Record<string, unknown[]>>;
export function countCycles(instrs: Array<Record<string, unknown[]>>): number;
