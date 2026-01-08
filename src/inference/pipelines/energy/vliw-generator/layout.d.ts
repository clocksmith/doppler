export class Op {
  constructor(engine: string, slot: unknown[], offloadable?: boolean, meta?: unknown);
  engine: string;
  slot: unknown[];
  offloadable: boolean;
  meta: unknown;
  id: number;
}
export function buildLayout(spec: Record<string, unknown>): unknown;
