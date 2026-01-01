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

// ============================================================================
// JSON Schema Definition (as TypeScript const for runtime validation)
// ============================================================================

/**
 * JSON Schema for adapter manifests.
 * Can be used with JSON Schema validators like Ajv.
 */
export const ADAPTER_MANIFEST_SCHEMA = {
  $schema: 'http://json-schema.org/draft-07/schema#',
  $id: 'https://doppler.dev/schemas/adapter-manifest.json',
  title: 'Adapter Manifest',
  description: 'Schema for LoRA adapter manifests in Doppler',
  type: 'object',
  required: ['id', 'name', 'baseModel', 'rank', 'alpha', 'targetModules'],
  properties: {
    id: {
      type: 'string',
      description: 'Unique identifier for the adapter (UUID or slug)',
      pattern: '^[a-zA-Z0-9_-]+$',
    },
    name: {
      type: 'string',
      description: 'Human-readable name for the adapter',
      minLength: 1,
      maxLength: 256,
    },
    version: {
      type: 'string',
      description: 'Semantic version of the adapter',
      pattern: '^\\d+\\.\\d+\\.\\d+(-[a-zA-Z0-9.]+)?$',
      default: '1.0.0',
    },
    description: {
      type: 'string',
      description: 'Detailed description of the adapter purpose',
      maxLength: 4096,
    },
    baseModel: {
      type: 'string',
      description: 'Identifier of the base model this adapter is trained for',
      examples: ['gemma-3-1b', 'llama-3-8b'],
    },
    rank: {
      type: 'integer',
      description: 'LoRA rank (dimensionality of the low-rank matrices)',
      minimum: 1,
      maximum: 1024,
    },
    alpha: {
      type: 'number',
      description: 'LoRA alpha scaling factor',
      minimum: 0.1,
    },
    targetModules: {
      type: 'array',
      description: 'List of modules this adapter modifies',
      items: {
        type: 'string',
        enum: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'gate_up_proj'],
      },
      minItems: 1,
      uniqueItems: true,
    },
    checksum: {
      type: 'string',
      description: 'SHA-256 or BLAKE3 hash of the weight file for integrity verification',
      pattern: '^[a-fA-F0-9]{64}$',
    },
    checksumAlgorithm: {
      type: 'string',
      description: 'Algorithm used for checksum',
      enum: ['sha256', 'blake3'],
      default: 'sha256',
    },
    weightsFormat: {
      type: 'string',
      description: 'Format of the weight tensors',
      enum: ['safetensors', 'npz', 'json', 'binary'],
      default: 'safetensors',
    },
    weightsPath: {
      type: 'string',
      description: 'Path or URL to the weights file (relative to manifest)',
    },
    weightsSize: {
      type: 'integer',
      description: 'Size of the weights file in bytes',
      minimum: 0,
    },
    tensors: {
      type: 'array',
      description: 'Inline tensor specifications (for small adapters)',
      items: {
        type: 'object',
        required: ['name', 'shape'],
        properties: {
          name: { type: 'string' },
          shape: {
            type: 'array',
            items: { type: 'integer' },
            minItems: 2,
            maxItems: 2,
          },
          dtype: {
            type: 'string',
            enum: ['f32', 'f16', 'bf16'],
            default: 'f32',
          },
          data: {
            type: 'array',
            items: { type: 'number' },
          },
          base64: { type: 'string' },
          opfsPath: { type: 'string' },
          url: { type: 'string' },
        },
      },
    },
    metadata: {
      type: 'object',
      description: 'Additional metadata about the adapter',
      properties: {
        author: { type: 'string' },
        license: { type: 'string' },
        tags: {
          type: 'array',
          items: { type: 'string' },
        },
        trainedOn: { type: 'string' },
        epochs: { type: 'number' },
        learningRate: { type: 'number' },
        createdAt: { type: 'string', format: 'date-time' },
        updatedAt: { type: 'string', format: 'date-time' },
      },
      additionalProperties: true,
    },
  },
  additionalProperties: false,
} as const;

// ============================================================================
// TypeScript Type Definitions
// ============================================================================

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

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Valid target module names.
 */
const VALID_TARGET_MODULES: readonly LoRAModuleName[] = [
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
  'gate_up_proj',
] as const;

/**
 * Validates an adapter manifest against the schema.
 */
export function validateManifest(manifest: unknown): ManifestValidationResult {
  const errors: ManifestValidationError[] = [];

  if (!manifest || typeof manifest !== 'object') {
    return { valid: false, errors: [{ field: 'root', message: 'Manifest must be an object' }] };
  }

  const m = manifest as Record<string, unknown>;

  // Required fields
  if (!m.id || typeof m.id !== 'string') {
    errors.push({ field: 'id', message: 'id is required and must be a string', value: m.id });
  } else if (!/^[a-zA-Z0-9_-]+$/.test(m.id)) {
    errors.push({ field: 'id', message: 'id must only contain alphanumeric characters, underscores, and hyphens', value: m.id });
  }

  if (!m.name || typeof m.name !== 'string') {
    errors.push({ field: 'name', message: 'name is required and must be a string', value: m.name });
  } else if (m.name.length > 256) {
    errors.push({ field: 'name', message: 'name must not exceed 256 characters', value: m.name });
  }

  if (!m.baseModel || typeof m.baseModel !== 'string') {
    errors.push({ field: 'baseModel', message: 'baseModel is required and must be a string', value: m.baseModel });
  }

  if (typeof m.rank !== 'number' || !Number.isInteger(m.rank) || m.rank < 1 || m.rank > 1024) {
    errors.push({ field: 'rank', message: 'rank must be an integer between 1 and 1024', value: m.rank });
  }

  if (typeof m.alpha !== 'number' || m.alpha < 0.1) {
    errors.push({ field: 'alpha', message: 'alpha must be a number >= 0.1', value: m.alpha });
  }

  if (!Array.isArray(m.targetModules) || m.targetModules.length === 0) {
    errors.push({ field: 'targetModules', message: 'targetModules must be a non-empty array', value: m.targetModules });
  } else {
    const uniqueModules = new Set<string>();
    for (const mod of m.targetModules) {
      if (!VALID_TARGET_MODULES.includes(mod as LoRAModuleName)) {
        errors.push({ field: 'targetModules', message: `Invalid target module: ${mod}`, value: mod });
      }
      if (uniqueModules.has(mod as string)) {
        errors.push({ field: 'targetModules', message: `Duplicate target module: ${mod}`, value: mod });
      }
      uniqueModules.add(mod as string);
    }
  }

  // Optional fields validation
  if (m.version !== undefined && typeof m.version === 'string') {
    if (!/^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$/.test(m.version)) {
      errors.push({ field: 'version', message: 'version must follow semantic versioning', value: m.version });
    }
  }

  if (m.checksum !== undefined && typeof m.checksum === 'string') {
    if (!/^[a-fA-F0-9]{64}$/.test(m.checksum)) {
      errors.push({ field: 'checksum', message: 'checksum must be a 64-character hex string', value: m.checksum });
    }
  }

  if (m.checksumAlgorithm !== undefined) {
    if (m.checksumAlgorithm !== 'sha256' && m.checksumAlgorithm !== 'blake3') {
      errors.push({ field: 'checksumAlgorithm', message: 'checksumAlgorithm must be sha256 or blake3', value: m.checksumAlgorithm });
    }
  }

  if (m.weightsFormat !== undefined) {
    const validFormats = ['safetensors', 'npz', 'json', 'binary'];
    if (!validFormats.includes(m.weightsFormat as string)) {
      errors.push({ field: 'weightsFormat', message: `weightsFormat must be one of: ${validFormats.join(', ')}`, value: m.weightsFormat });
    }
  }

  if (m.tensors !== undefined && Array.isArray(m.tensors)) {
    for (let i = 0; i < m.tensors.length; i++) {
      const tensor = m.tensors[i] as Record<string, unknown>;
      if (!tensor.name || typeof tensor.name !== 'string') {
        errors.push({ field: `tensors[${i}].name`, message: 'tensor name is required', value: tensor.name });
      }
      if (!Array.isArray(tensor.shape) || tensor.shape.length !== 2) {
        errors.push({ field: `tensors[${i}].shape`, message: 'tensor shape must be [rows, cols]', value: tensor.shape });
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Parses and validates a manifest from JSON string.
 */
export function parseManifest(json: string): AdapterManifest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (e) {
    throw new Error(`Invalid JSON: ${(e as Error).message}`);
  }

  const validation = validateManifest(parsed);
  if (!validation.valid) {
    const errorMessages = validation.errors.map(e => `${e.field}: ${e.message}`).join('; ');
    throw new Error(`Manifest validation failed: ${errorMessages}`);
  }

  return parsed as AdapterManifest;
}

/**
 * Serializes an adapter manifest to JSON string.
 */
export function serializeManifest(manifest: AdapterManifest, pretty = false): string {
  const validation = validateManifest(manifest);
  if (!validation.valid) {
    const errorMessages = validation.errors.map(e => `${e.field}: ${e.message}`).join('; ');
    throw new Error(`Cannot serialize invalid manifest: ${errorMessages}`);
  }

  return JSON.stringify(manifest, null, pretty ? 2 : undefined);
}

/**
 * Creates a minimal valid manifest with defaults.
 */
export function createManifest(
  options: MinimalAdapterManifest & Partial<AdapterManifest>
): AdapterManifest {
  return {
    version: '1.0.0',
    checksumAlgorithm: 'sha256',
    weightsFormat: 'safetensors',
    ...options,
  };
}

/**
 * Computes the expected scale factor from rank and alpha.
 */
export function computeLoRAScale(rank: number, alpha: number): number {
  return rank > 0 ? alpha / rank : 1;
}
