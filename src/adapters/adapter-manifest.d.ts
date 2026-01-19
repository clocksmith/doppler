/**
 * Adapter Manifest Format Definition
 *
 * Defines the JSON schema and TypeScript types for adapter manifests.
 * This enables self-describing adapters with versioning, checksum validation,
 * and compatibility checking.
 *
 * @module adapters/adapter-manifest
 */

import type { LoRAModuleName } from '../inference/pipeline/lora-types.js';

export declare const DEFAULT_ADAPTER_MANIFEST_DEFAULTS: {
  readonly version: '1.0.0';
  readonly checksumAlgorithm: 'sha256';
  readonly weightsFormat: 'safetensors';
  readonly tensorDtype: 'f32';
};

/**
 * JSON Schema for adapter manifests.
 * Can be used with JSON Schema validators like Ajv.
 */
export declare const ADAPTER_MANIFEST_SCHEMA: {
  readonly $schema: 'http://json-schema.org/draft-07/schema#';
  readonly $id: 'https://doppler.dev/schemas/adapter-manifest.json';
  readonly title: 'Adapter Manifest';
  readonly description: 'Schema for LoRA adapter manifests in Doppler';
  readonly type: 'object';
  readonly required: readonly ['id', 'name', 'baseModel', 'rank', 'alpha', 'targetModules'];
  readonly properties: {
    readonly id: {
      readonly type: 'string';
      readonly description: 'Unique identifier for the adapter (UUID or slug)';
      readonly pattern: '^[a-zA-Z0-9_-]+$';
    };
    readonly name: {
      readonly type: 'string';
      readonly description: 'Human-readable name for the adapter';
      readonly minLength: 1;
      readonly maxLength: 256;
    };
    readonly version: {
      readonly type: 'string';
      readonly description: 'Semantic version of the adapter';
      readonly pattern: '^\\d+\\.\\d+\\.\\d+(-[a-zA-Z0-9.]+)?$';
      readonly default: '1.0.0';
    };
    readonly description: {
      readonly type: 'string';
      readonly description: 'Detailed description of the adapter purpose';
      readonly maxLength: 4096;
    };
    readonly baseModel: {
      readonly type: 'string';
      readonly description: 'Identifier of the base model this adapter is trained for';
      readonly examples: readonly ['gemma-3-1b', 'llama-3-8b'];
    };
    readonly rank: {
      readonly type: 'integer';
      readonly description: 'LoRA rank (dimensionality of the low-rank matrices)';
      readonly minimum: 1;
      readonly maximum: 1024;
    };
    readonly alpha: {
      readonly type: 'number';
      readonly description: 'LoRA alpha scaling factor';
      readonly minimum: 0.1;
    };
    readonly targetModules: {
      readonly type: 'array';
      readonly description: 'List of modules this adapter modifies';
      readonly items: {
        readonly type: 'string';
        readonly enum: readonly ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'gate_up_proj'];
      };
      readonly minItems: 1;
      readonly uniqueItems: true;
    };
    readonly checksum: {
      readonly type: 'string';
      readonly description: 'SHA-256 or BLAKE3 hash of the weight file for integrity verification';
      readonly pattern: '^[a-fA-F0-9]{64}$';
    };
    readonly checksumAlgorithm: {
      readonly type: 'string';
      readonly description: 'Algorithm used for checksum';
      readonly enum: readonly ['sha256', 'blake3'];
      readonly default: 'sha256';
    };
    readonly weightsFormat: {
      readonly type: 'string';
      readonly description: 'Format of the weight tensors';
      readonly enum: readonly ['safetensors', 'npz', 'json', 'binary'];
      readonly default: 'safetensors';
    };
    readonly weightsPath: {
      readonly type: 'string';
      readonly description: 'Path or URL to the weights file (relative to manifest)';
    };
    readonly weightsSize: {
      readonly type: 'integer';
      readonly description: 'Size of the weights file in bytes';
      readonly minimum: 0;
    };
    readonly tensors: {
      readonly type: 'array';
      readonly description: 'Inline tensor specifications (for small adapters)';
      readonly items: {
        readonly type: 'object';
        readonly required: readonly ['name', 'shape'];
        readonly properties: {
          readonly name: { readonly type: 'string' };
          readonly shape: {
            readonly type: 'array';
            readonly items: { readonly type: 'integer' };
            readonly minItems: 2;
            readonly maxItems: 2;
          };
          readonly dtype: {
            readonly type: 'string';
            readonly enum: readonly ['f32', 'f16', 'bf16'];
            readonly default: 'f32';
          };
          readonly data: {
            readonly type: 'array';
            readonly items: { readonly type: 'number' };
          };
          readonly base64: { readonly type: 'string' };
          readonly opfsPath: { readonly type: 'string' };
          readonly url: { readonly type: 'string' };
        };
      };
    };
    readonly metadata: {
      readonly type: 'object';
      readonly description: 'Additional metadata about the adapter';
      readonly properties: {
        readonly author: { readonly type: 'string' };
        readonly license: { readonly type: 'string' };
        readonly tags: {
          readonly type: 'array';
          readonly items: { readonly type: 'string' };
        };
        readonly trainedOn: { readonly type: 'string' };
        readonly epochs: { readonly type: 'number' };
        readonly learningRate: { readonly type: 'number' };
        readonly createdAt: { readonly type: 'string'; readonly format: 'date-time' };
        readonly updatedAt: { readonly type: 'string'; readonly format: 'date-time' };
      };
      readonly additionalProperties: true;
    };
  };
  readonly additionalProperties: false;
};

/**
 * Tensor specification for inline weight data.
 */
export interface AdapterTensorSpec {
  /** Tensor name following pattern: layer.{N}.{module}.lora_{a|b} */
  name: string;
  /** Shape as [rows, cols] */
  shape: [number, number];
  /** Data type (default: f32) */
  dtype?: 'f32' | 'f16' | 'bf16';
  /** Inline data as number array */
  data?: number[];
  /** Base64-encoded binary data */
  base64?: string;
  /** Path in OPFS storage */
  opfsPath?: string;
  /** URL to fetch tensor data from */
  url?: string;
}

/**
 * Adapter metadata for tracking provenance.
 */
export interface AdapterMetadata {
  /** Author or organization */
  author?: string;
  /** License identifier (e.g., 'MIT', 'Apache-2.0') */
  license?: string;
  /** Tags for categorization */
  tags?: string[];
  /** Description of training data */
  trainedOn?: string;
  /** Number of training epochs */
  epochs?: number;
  /** Training learning rate */
  learningRate?: number;
  /** Creation timestamp (ISO 8601) */
  createdAt?: string;
  /** Last update timestamp (ISO 8601) */
  updatedAt?: string;
  /** Additional custom metadata */
  [key: string]: unknown;
}

/**
 * Full adapter manifest structure.
 * This is the primary type for adapter definitions.
 */
export interface AdapterManifest {
  /** Unique identifier (UUID or slug) */
  id: string;
  /** Human-readable name */
  name: string;
  /** Semantic version (default: '1.0.0') */
  version?: string;
  /** Detailed description */
  description?: string;
  /** Base model identifier this adapter is compatible with */
  baseModel: string;
  /** LoRA rank (dimensionality) */
  rank: number;
  /** LoRA alpha scaling factor */
  alpha: number;
  /** List of modules this adapter modifies */
  targetModules: LoRAModuleName[];
  /** Content checksum for integrity verification */
  checksum?: string;
  /** Algorithm used for checksum */
  checksumAlgorithm?: 'sha256' | 'blake3';
  /** Format of weight tensors */
  weightsFormat?: 'safetensors' | 'npz' | 'json' | 'binary';
  /** Path or URL to weights file */
  weightsPath?: string;
  /** Size of weights file in bytes */
  weightsSize?: number;
  /** Inline tensor specifications */
  tensors?: AdapterTensorSpec[];
  /** Additional metadata */
  metadata?: AdapterMetadata;
}

/**
 * Minimal adapter manifest with only required fields.
 */
export type MinimalAdapterManifest = Pick<
  AdapterManifest,
  'id' | 'name' | 'baseModel' | 'rank' | 'alpha' | 'targetModules'
>;

/**
 * Adapter manifest validation result.
 */
export interface ManifestValidationResult {
  valid: boolean;
  errors: ManifestValidationError[];
}

/**
 * Validation error details.
 */
export interface ManifestValidationError {
  field: string;
  message: string;
  value?: unknown;
}

/**
 * Validates an adapter manifest against the schema.
 */
export declare function validateManifest(manifest: unknown): ManifestValidationResult;

export declare function applyAdapterManifestDefaults<T extends Record<string, unknown>>(manifest: T): T;

/**
 * Parses and validates a manifest from JSON string.
 */
export declare function parseManifest(json: string): AdapterManifest;

/**
 * Serializes an adapter manifest to JSON string.
 */
export declare function serializeManifest(manifest: AdapterManifest, pretty?: boolean): string;

/**
 * Creates a minimal valid manifest with defaults.
 */
export declare function createManifest(
  options: MinimalAdapterManifest & Partial<AdapterManifest>
): AdapterManifest;

/**
 * Computes the expected scale factor from rank and alpha.
 */
export declare function computeLoRAScale(rank: number, alpha: number): number;
