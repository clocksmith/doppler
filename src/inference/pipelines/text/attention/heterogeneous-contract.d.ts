export function queryScaleValidationError(value: unknown): string | null;
export function resolveQueryScale(value: unknown): number;
export function ropeDisabledLayersValidationError(value: unknown): string | null;
export function appendHeterogeneousAttentionValidation(
  errors: string[],
  inf: { attention: { queryScale?: unknown }; rope: { disabledLayers?: unknown } }
): void;
export function resolveHeterogeneousAttentionContract(
  inf: { attention: { queryScale?: unknown }; rope: { disabledLayers?: unknown } },
  numLayers: number,
  modelId: string
): { queryScale: number; ropeDisabledLayers: number[] };
export function isRoPEDisabledForLayer(
  config: { ropeDisabledLayers?: number[] | null } | null,
  layerIdx: number
): boolean;
