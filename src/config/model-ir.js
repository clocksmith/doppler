/**
 * Doppler ModelIR: hardware-independent facts extracted from a source artifact.
 *
 * ModelIR is intentionally fail-closed. Callers must provide every semantic
 * field; this module never fills an unknown model fact with a plausible value.
 *
 * @module config/model-ir
 */

import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const MODEL_IR_SCHEMA_ID = 'doppler.model-ir/v1';
export const MODEL_IR_SCHEMA_VERSION = 1;

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) {
    errors.push(`${label} must be a non-empty string.`);
  }
}

function requirePositiveInteger(value, label, errors) {
  if (!Number.isInteger(value) || value < 1) {
    errors.push(`${label} must be a positive integer.`);
  }
}

/**
 * Validates a hardware-independent semantic graph.
 *
 * @param {unknown} ir
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validateModelIR(ir) {
  const errors = [];
  if (!isObject(ir)) {
    return { ok: false, errors: ['ModelIR must be a non-null object.'] };
  }
  if (ir.schema !== MODEL_IR_SCHEMA_ID) {
    errors.push(`schema must be "${MODEL_IR_SCHEMA_ID}", received "${ir.schema}".`);
  }
  if (ir.schemaVersion !== MODEL_IR_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${MODEL_IR_SCHEMA_VERSION}, received ${ir.schemaVersion}.`);
  }
  requireString(ir.modelId, 'modelId', errors);
  requireString(ir.architecture, 'architecture', errors);
  requirePositiveInteger(ir.vocabSize, 'vocabSize', errors);
  requirePositiveInteger(ir.hiddenSize, 'hiddenSize', errors);
  requirePositiveInteger(ir.numLayers, 'numLayers', errors);

  if (!isObject(ir.sourceIdentity)) {
    errors.push('sourceIdentity must be an object.');
  } else {
    requireString(ir.sourceIdentity.manifestArtifactId, 'sourceIdentity.manifestArtifactId', errors);
    if (!SHA256_PATTERN.test(ir.sourceIdentity.manifestHash || '')) {
      errors.push('sourceIdentity.manifestHash must be a SHA-256 digest.');
    }
  }

  if (!isObject(ir.tensorRoles) || Object.keys(ir.tensorRoles).length === 0) {
    errors.push('tensorRoles must be a non-empty object derived from source tensors.');
  } else {
    for (const [name, tensor] of Object.entries(ir.tensorRoles)) {
      if (!isObject(tensor)) {
        errors.push(`tensorRoles.${name} must be an object.`);
        continue;
      }
      requireString(tensor.role, `tensorRoles.${name}.role`, errors);
      if (!Array.isArray(tensor.shape) || tensor.shape.some((dimension) => !Number.isInteger(dimension) || dimension < 1)) {
        errors.push(`tensorRoles.${name}.shape must contain positive integer dimensions.`);
      }
      requireString(tensor.semanticDtype, `tensorRoles.${name}.semanticDtype`, errors);
    }
  }

  if (!Array.isArray(ir.layers) || ir.layers.length !== ir.numLayers) {
    errors.push('layers must contain exactly numLayers entries.');
  } else {
    const indices = new Set();
    for (const [position, layer] of ir.layers.entries()) {
      if (!isObject(layer)) {
        errors.push(`layers[${position}] must be an object.`);
        continue;
      }
      if (!Number.isInteger(layer.index) || layer.index < 0 || indices.has(layer.index)) {
        errors.push(`layers[${position}].index must be a unique non-negative integer.`);
      }
      indices.add(layer.index);
      requireString(layer.type, `layers[${position}].type`, errors);
    }
  }

  if (!isObject(ir.attentionGeometry)) {
    errors.push('attentionGeometry must be an object.');
  } else {
    requirePositiveInteger(ir.attentionGeometry.numHeads, 'attentionGeometry.numHeads', errors);
    requirePositiveInteger(ir.attentionGeometry.numKvHeads, 'attentionGeometry.numKvHeads', errors);
    requirePositiveInteger(ir.attentionGeometry.headDim, 'attentionGeometry.headDim', errors);
  }
  if (!isObject(ir.normalization)) {
    errors.push('normalization must be an object.');
  } else {
    requireString(ir.normalization.type, 'normalization.type', errors);
    if (!Number.isFinite(ir.normalization.eps) || ir.normalization.eps <= 0) {
      errors.push('normalization.eps must be a positive finite number.');
    }
  }
  if (ir.rope !== null) {
    if (!isObject(ir.rope)) {
      errors.push('rope must be an object or null.');
    } else {
      requirePositiveInteger(ir.rope.dimension, 'rope.dimension', errors);
      if (!Number.isFinite(ir.rope.baseFreq) || ir.rope.baseFreq < 1) {
        errors.push('rope.baseFreq must be a finite number greater than or equal to one.');
      }
    }
  }
  if (!isObject(ir.ffn)) {
    errors.push('ffn must be an object.');
  } else {
    requireString(ir.ffn.type, 'ffn.type', errors);
    requirePositiveInteger(ir.ffn.intermediateSize, 'ffn.intermediateSize', errors);
  }
  if (!isObject(ir.outputTopology)) {
    errors.push('outputTopology must be an object.');
  } else {
    requireString(ir.outputTopology.headType, 'outputTopology.headType', errors);
    if (typeof ir.outputTopology.tieWeights !== 'boolean') {
      errors.push('outputTopology.tieWeights must be boolean.');
    }
  }
  if (!Array.isArray(ir.phases) || ir.phases.length === 0) {
    errors.push('phases must be a non-empty array.');
  } else if (ir.phases.some((phase) => typeof phase !== 'string' || !phase.trim())) {
    errors.push('phases must contain non-empty strings.');
  }

  return { ok: errors.length === 0, errors };
}

/**
 * @param {unknown} ir
 * @returns {`sha256:${string}`}
 */
export function hashModelIR(ir) {
  const validation = validateModelIR(ir);
  if (!validation.ok) {
    throw new Error(`Cannot hash invalid ModelIR: ${validation.errors.join('; ')}`);
  }
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(ir)))}`;
}

/**
 * Creates a ModelIR without supplying semantic defaults.
 *
 * @param {object} params
 * @returns {object}
 */
export function createModelIR(params) {
  if (!isObject(params)) {
    throw new Error('createModelIR requires an object.');
  }
  const ir = {
    schema: MODEL_IR_SCHEMA_ID,
    schemaVersion: MODEL_IR_SCHEMA_VERSION,
    modelId: params.modelId,
    architecture: params.architecture,
    vocabSize: params.vocabSize,
    hiddenSize: params.hiddenSize,
    numLayers: params.numLayers,
    sourceIdentity: params.sourceIdentity,
    tensorRoles: params.tensorRoles,
    layers: params.layers,
    attentionGeometry: params.attentionGeometry,
    normalization: params.normalization,
    rope: params.rope ?? null,
    ffn: params.ffn,
    outputTopology: params.outputTopology,
    phases: params.phases,
  };
  const validation = validateModelIR(ir);
  if (!validation.ok) {
    throw new Error(`Failed to create valid ModelIR: ${validation.errors.join('; ')}`);
  }
  return ir;
}
