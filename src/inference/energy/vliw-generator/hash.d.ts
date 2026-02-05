export const HASH_STAGES: Array<[string, number, string, string, number]>;
export function splitHashStages(): {
  linear: Array<{ mult: number; add: number }>;
  bitwise: Array<{ op1: string; const: number; op2: string; shift_op: string; shift: number }>;
};
