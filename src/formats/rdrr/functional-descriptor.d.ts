export interface FunctionalDescriptorProofGate {
  sensitivity?: string;
  compression?: string;
  determinism?: string;
}

export interface FunctionalDescriptorManifest {
  schema_version: 'manifoldgguf.v0.1';
  tensor_name?: string;
  source_shape?: [number, number];
  slice_shape: [number, number];
  crop_shape?: [number, number];
  padded_shape?: [number, number];
  tile_shape?: [number, number];
  padding?: {
    rows?: number;
    cols?: number;
  };
  storage_type: 'functional_descriptor';
  descriptor_hash?: string;
  descriptor_bytes?: number;
  dense_f16_bytes?: number;
  compression_ratio?: number;
  proof_status?: string;
  proof_status_gate?: FunctionalDescriptorProofGate;
  components: {
    prng_substrate: {
      algorithm: 'coord_hash_normal_v1';
      seed: number;
      learned_scale: number;
      learned_scale_frozen?: boolean;
    };
    kronecker_sum: {
      shard_file: string;
      shard_hash?: string;
      rank_terms?: number;
      factor_shapes?: unknown[];
    };
    coordinate_inr: {
      shard_file: string;
      shard_hash?: string;
      type: 'siren';
      network_dims?: number[];
      omega_0?: number;
    };
    sparse_outliers: {
      shard_file: string;
      shard_hash?: string;
      format?: 'coo_v1';
      value_dtype?: string;
      actual_nnz?: number;
    };
  };
}

export interface FunctionalDescriptorValidationResult {
  valid: boolean;
  errors: string[];
}

export declare function isFunctionalDescriptorDtype(dtype: unknown): boolean;

export declare function getFunctionalDescriptorManifest(info: unknown): FunctionalDescriptorManifest | Record<string, unknown> | null;

export declare function validateFunctionalDescriptorManifest(
  value: unknown,
  label?: string
): FunctionalDescriptorValidationResult;

export declare function assertFunctionalDescriptorManifest(
  value: unknown,
  label?: string
): FunctionalDescriptorManifest;
