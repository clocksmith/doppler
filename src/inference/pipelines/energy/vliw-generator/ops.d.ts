export function buildOps(
  spec: Record<string, unknown>,
  layout: unknown,
  orderedOps?: unknown[],
): {
  valu_ops: unknown[];
  alu_ops: unknown[];
  flow_ops: unknown[];
  load_ops: unknown[];
  store_ops: unknown[];
};
