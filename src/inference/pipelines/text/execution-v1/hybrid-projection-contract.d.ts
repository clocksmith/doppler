import type { ExecutionV1GraphSchema } from "../../../../config/schema/execution/execution-v1.schema.js";
export function assertHybridLinearProjectionIsolation(
  execution: ExecutionV1GraphSchema, layerTypes: readonly string[] | null | undefined,
  modelId: string, weightDtype: string | null | undefined
): void;
