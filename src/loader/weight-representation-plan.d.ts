import type { TensorLocation } from './loader-types.js';
import type { KernelPathSchema } from '../config/schema/kernel-path.schema.js';
export interface WeightConsumer {
  readonly tensor: string;
  readonly role: string;
  readonly layer: number;
  readonly phase: string;
  readonly path: string;
  readonly variant: string | null;
  readonly declaredWeightDtype: string | null;
  readonly weightDtype: string | null;
}
export interface WeightRepresentationRequirement {
  readonly mode: 'packed-only' | 'dense-only' | 'both';
  readonly complete: boolean;
  readonly primaryDtype: 'f16' | null;
  readonly dtypes: readonly string[];
  readonly aliases: readonly string[];
  readonly consumers: readonly WeightConsumer[];
  readonly keepF32Weights: boolean;
}
export type WeightRepresentationPlan = Readonly<Record<string, WeightRepresentationRequirement>>;
export function compileWeightRepresentationPlan(
  locations: ReadonlyMap<string, TensorLocation>,
  manifest: { architecture: unknown; inference?: { output?: { tieWordEmbeddings?: boolean } } },
  paths: readonly KernelPathSchema[],
  defaults: { useFusedQ4K: boolean; keepF32Weights: boolean; q4kMaterializationMode: string; q4kLayout: string | null }
): WeightRepresentationPlan;
