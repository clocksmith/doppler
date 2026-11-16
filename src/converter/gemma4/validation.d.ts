import type { ManifestInferenceSchema, ParsedModelSchema, TensorInfoSchema } from '../../config/schema/index.js';
export function validateInt4PleMaterializationContract(
  tensorLocations: Record<string, { sourceTransform?: { kind?: string; sourceDtype?: string } | null }> | null | undefined,
  inference: ManifestInferenceSchema | null | undefined,
  modelId: string
): void;
export function assertSupportedGemma4Conversion(
  model: Pick<ParsedModelSchema, 'config'> | null | undefined,
  tensors: readonly TensorInfoSchema[] | null | undefined,
  modelId: string
): void;
