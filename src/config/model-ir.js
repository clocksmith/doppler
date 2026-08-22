/**
 * Doppler ModelIR: Hardware-Agnostic Semantic Computation Graph
 *
 * @module config/model-ir
 */

import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const MODEL_IR_SCHEMA_ID = 'doppler.model-ir/v1';
export const MODEL_IR_SCHEMA_VERSION = 1;

/**
 * Validates that an object conforms to the ModelIR contract.
 *
 * @param {object} ir
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validateModelIR(ir) {
  const errors = [];
  if (!ir || typeof ir !== 'object' || Array.isArray(ir)) {
    return { ok: false, errors: ['ModelIR must be a non-null object.'] };
  }
  if (ir.schema !== MODEL_IR_SCHEMA_ID) {
    errors.push(`schema must be "${MODEL_IR_SCHEMA_ID}", received "${ir.schema}".`);
  }
  if (ir.schemaVersion !== MODEL_IR_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${MODEL_IR_SCHEMA_VERSION}, received ${ir.schemaVersion}.`);
  }
  if (typeof ir.modelId !== 'string' || !ir.modelId.trim()) {
    errors.push('modelId must be a non-empty string.');
  }
  if (typeof ir.architecture !== 'string' || !ir.architecture.trim()) {
    errors.push('architecture must be a non-empty string.');
  }
  if (!ir.tensorRoles || typeof ir.tensorRoles !== 'object') {
    errors.push('tensorRoles must be an object.');
  }
  if (!Array.isArray(ir.layers) || ir.layers.length === 0) {
    errors.push('layers must be a non-empty array.');
  }
  if (!ir.attentionGeometry || typeof ir.attentionGeometry !== 'object') {
    errors.push('attentionGeometry must be an object.');
  }
  if (!ir.normalization || typeof ir.normalization !== 'object') {
    errors.push('normalization must be an object.');
  }
  if (!ir.ffn || typeof ir.ffn !== 'object') {
    errors.push('ffn must be an object.');
  }
  if (!ir.outputTopology || typeof ir.outputTopology !== 'object') {
    errors.push('outputTopology must be an object.');
  }
  if (!Array.isArray(ir.phases) || ir.phases.length === 0) {
    errors.push('phases must be a non-empty array.');
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}

/**
 * Canonical cryptographic digest of a ModelIR object.
 *
 * @param {object} ir
 * @returns {`sha256:${string}`}
 */
export function hashModelIR(ir) {
  const validation = validateModelIR(ir);
  if (!validation.ok) {
    throw new Error(`Cannot hash invalid ModelIR: ${validation.errors.join('; ')}`);
  }
  const canonicalJson = JSON.stringify(stableSortObject(ir));
  return `sha256:${sha256Hex(canonicalJson)}`;
}

/**
 * Creates and validates a normalized ModelIR structure.
 *
 * @param {object} params
 * @returns {object} ModelIR
 */
export function createModelIR(params) {
  const numLayers = Number(params.numLayers || 0);
  const defaultLayers = numLayers > 0
    ? Array.from({ length: numLayers }, (_, index) => ({ index, type: 'transformer' }))
    : [{ index: 0, type: 'transformer' }];

  const ir = {
    schema: MODEL_IR_SCHEMA_ID,
    schemaVersion: MODEL_IR_SCHEMA_VERSION,
    modelId: String(params.modelId || '').trim(),
    architecture: String(params.architecture || '').trim(),
    vocabSize: Number(params.vocabSize || 0),
    hiddenSize: Number(params.hiddenSize || 0),
    numLayers: numLayers || defaultLayers.length,
    tensorRoles: params.tensorRoles || {},
    layers: Array.isArray(params.layers) && params.layers.length > 0 ? params.layers : defaultLayers,
    attentionGeometry: params.attentionGeometry || { numHeads: 1, numKvHeads: 1, headDim: 64 },
    normalization: params.normalization || { type: 'rmsnorm', eps: 1e-6 },
    rope: params.rope ?? null,
    ffn: params.ffn || { type: 'swiglu', intermediateSize: 2048 },
    outputTopology: params.outputTopology || { headType: 'causal-lm', tieWeights: false },
    phases: Array.isArray(params.phases) ? params.phases : ['prefill', 'decode'],
  };

  const validation = validateModelIR(ir);
  if (!validation.ok) {
    throw new Error(`Failed to create valid ModelIR: ${validation.errors.join('; ')}`);
  }
  return ir;
}
