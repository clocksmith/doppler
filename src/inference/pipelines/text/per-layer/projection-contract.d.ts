export { resolveDensePleProjectionWeight } from "./per-layer-inputs.js";
export function resolvePleActivationStaticQdqConfig(weights: {
  perLayerModelProjectionInputActivationStaticScale?: unknown;
  perLayerModelProjectionOutputActivationStaticScale?: unknown;
} | null | undefined): { inputScale: number; outputScale: number } | null;
