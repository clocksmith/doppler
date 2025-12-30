export interface BackwardRegistryOpSchema {
  backward: string;
  grads: string[];
  requires_transpose?: boolean;
  notes?: string;
}

export interface BackwardRegistrySchema {
  ops: Record<string, BackwardRegistryOpSchema>;
}

export declare function validateBackwardRegistry(
  registry: BackwardRegistrySchema
): BackwardRegistrySchema;
