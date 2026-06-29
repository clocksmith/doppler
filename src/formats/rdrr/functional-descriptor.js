const DESCRIPTOR_SCHEMA_VERSION = 'manifoldgguf.v0.1';
const DESCRIPTOR_STORAGE_TYPE = 'functional_descriptor';
const DESCRIPTOR_PRNG_ALGORITHM = 'coord_hash_normal_v1';
const DESCRIPTOR_COORDINATE_INR_TYPE = 'siren';
const DESCRIPTOR_SPARSE_FORMAT = 'coo_v1';

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function isNonNegativeInteger(value) {
  return Number.isInteger(value) && value >= 0;
}

function validateShape2(errors, value, label) {
  if (!Array.isArray(value) || value.length !== 2) {
    errors.push(`${label} must be [rows, cols]`);
    return null;
  }
  const rows = Number(value[0]);
  const cols = Number(value[1]);
  if (!isPositiveInteger(rows) || !isPositiveInteger(cols)) {
    errors.push(`${label} must contain positive integers`);
    return null;
  }
  return [rows, cols];
}

function validateOptionalShape2(errors, manifest, field, label) {
  if (manifest[field] === undefined) {
    return null;
  }
  return validateShape2(errors, manifest[field], `${label}.${field}`);
}

function validateOptionalNonNegativeInteger(errors, object, field, label) {
  if (object[field] === undefined) {
    return;
  }
  if (!isNonNegativeInteger(Number(object[field]))) {
    errors.push(`${label}.${field} must be a non-negative integer`);
  }
}

function validateOptionalPositiveInteger(errors, object, field, label) {
  if (object[field] === undefined) {
    return;
  }
  if (!isPositiveInteger(Number(object[field]))) {
    errors.push(`${label}.${field} must be a positive integer`);
  }
}

function validateOptionalFiniteNumber(errors, object, field, label) {
  if (object[field] === undefined) {
    return;
  }
  if (!Number.isFinite(Number(object[field]))) {
    errors.push(`${label}.${field} must be a finite number`);
  }
}

function validateOptionalSha256(errors, object, field, label) {
  if (object[field] === undefined) {
    return;
  }
  if (typeof object[field] !== 'string' || !/^sha256:[a-f0-9]{64}$/i.test(object[field].trim())) {
    errors.push(`${label}.${field} must be sha256:<64 hex chars>`);
  }
}

function validateShardFile(errors, component, fieldLabel) {
  if (!isPlainObject(component)) {
    errors.push(`${fieldLabel} must be an object`);
    return;
  }
  if (typeof component.shard_file !== 'string' || component.shard_file.trim().length === 0) {
    errors.push(`${fieldLabel}.shard_file must be a non-empty string`);
  }
  validateOptionalSha256(errors, component, 'shard_hash', fieldLabel);
}

function validateProofGate(errors, manifest, label) {
  if (manifest.proof_status !== undefined) {
    if (typeof manifest.proof_status !== 'string' || manifest.proof_status.trim().length === 0) {
      errors.push(`${label}.proof_status must be a non-empty string when present`);
    }
  }
  if (manifest.proof_status_gate === undefined) {
    return;
  }
  if (!isPlainObject(manifest.proof_status_gate)) {
    errors.push(`${label}.proof_status_gate must be an object`);
    return;
  }
  for (const field of ['sensitivity', 'compression', 'determinism']) {
    const value = manifest.proof_status_gate[field];
    if (value === undefined) {
      continue;
    }
    if (typeof value !== 'string' || value.trim().length === 0) {
      errors.push(`${label}.proof_status_gate.${field} must be a non-empty string when present`);
    }
  }
}

export function isFunctionalDescriptorDtype(dtype) {
  return String(dtype || '').trim().toUpperCase() === 'FUNCTIONAL_DESCRIPTOR';
}

export function getFunctionalDescriptorManifest(info) {
  return info?.descriptorManifest
    ?? info?.descriptor_manifest
    ?? info?.functionalDescriptor
    ?? info?.functional_descriptor
    ?? null;
}

export function validateFunctionalDescriptorManifest(value, label = 'descriptorManifest') {
  const errors = [];
  if (!isPlainObject(value)) {
    return {
      valid: false,
      errors: [`${label} must be an object`],
    };
  }

  if (value.schema_version !== DESCRIPTOR_SCHEMA_VERSION) {
    errors.push(`${label}.schema_version must be "${DESCRIPTOR_SCHEMA_VERSION}"`);
  }
  if (value.storage_type !== DESCRIPTOR_STORAGE_TYPE) {
    errors.push(`${label}.storage_type must be "${DESCRIPTOR_STORAGE_TYPE}"`);
  }

  const sliceShape = validateShape2(errors, value.slice_shape, `${label}.slice_shape`);
  const cropShape = validateOptionalShape2(errors, value, 'crop_shape', label);
  const paddedShape = validateOptionalShape2(errors, value, 'padded_shape', label);
  validateOptionalShape2(errors, value, 'source_shape', label);
  validateOptionalShape2(errors, value, 'tile_shape', label);

  const runtimeShape = paddedShape ?? sliceShape;
  const effectiveCropShape = cropShape ?? sliceShape;
  if (runtimeShape && effectiveCropShape) {
    if (effectiveCropShape[0] > runtimeShape[0] || effectiveCropShape[1] > runtimeShape[1]) {
      errors.push(`${label}.crop_shape must not exceed reconstruction shape`);
    }
  }

  if (isPlainObject(value.padding)) {
    validateOptionalNonNegativeInteger(errors, value.padding, 'rows', `${label}.padding`);
    validateOptionalNonNegativeInteger(errors, value.padding, 'cols', `${label}.padding`);
  } else if (value.padding !== undefined) {
    errors.push(`${label}.padding must be an object`);
  }

  validateOptionalSha256(errors, value, 'descriptor_hash', label);
  validateOptionalPositiveInteger(errors, value, 'descriptor_bytes', label);
  validateOptionalPositiveInteger(errors, value, 'dense_f16_bytes', label);
  validateOptionalFiniteNumber(errors, value, 'compression_ratio', label);
  validateProofGate(errors, value, label);

  const components = value.components;
  if (!isPlainObject(components)) {
    errors.push(`${label}.components must be an object`);
    return { valid: errors.length === 0, errors };
  }

  const prng = components.prng_substrate;
  if (!isPlainObject(prng)) {
    errors.push(`${label}.components.prng_substrate must be an object`);
  } else {
    if (prng.algorithm !== DESCRIPTOR_PRNG_ALGORITHM) {
      errors.push(`${label}.components.prng_substrate.algorithm must be "${DESCRIPTOR_PRNG_ALGORITHM}"`);
    }
    if (!Number.isInteger(Number(prng.seed))) {
      errors.push(`${label}.components.prng_substrate.seed must be an integer`);
    }
    if (!Number.isFinite(Number(prng.learned_scale))) {
      errors.push(`${label}.components.prng_substrate.learned_scale must be a finite number`);
    }
    if (prng.learned_scale_frozen !== undefined && typeof prng.learned_scale_frozen !== 'boolean') {
      errors.push(`${label}.components.prng_substrate.learned_scale_frozen must be boolean when present`);
    }
  }

  validateShardFile(errors, components.kronecker_sum, `${label}.components.kronecker_sum`);
  if (isPlainObject(components.kronecker_sum)) {
    validateOptionalPositiveInteger(errors, components.kronecker_sum, 'rank_terms', `${label}.components.kronecker_sum`);
  }

  validateShardFile(errors, components.coordinate_inr, `${label}.components.coordinate_inr`);
  if (isPlainObject(components.coordinate_inr)) {
    if (components.coordinate_inr.type !== DESCRIPTOR_COORDINATE_INR_TYPE) {
      errors.push(`${label}.components.coordinate_inr.type must be "${DESCRIPTOR_COORDINATE_INR_TYPE}"`);
    }
    if (components.coordinate_inr.network_dims !== undefined) {
      if (!Array.isArray(components.coordinate_inr.network_dims) ||
          components.coordinate_inr.network_dims.some((dim) => !isPositiveInteger(Number(dim)))) {
        errors.push(`${label}.components.coordinate_inr.network_dims must contain positive integers`);
      }
    }
    validateOptionalFiniteNumber(errors, components.coordinate_inr, 'omega_0', `${label}.components.coordinate_inr`);
  }

  validateShardFile(errors, components.sparse_outliers, `${label}.components.sparse_outliers`);
  if (isPlainObject(components.sparse_outliers)) {
    if (components.sparse_outliers.format !== undefined &&
        components.sparse_outliers.format !== DESCRIPTOR_SPARSE_FORMAT) {
      errors.push(`${label}.components.sparse_outliers.format must be "${DESCRIPTOR_SPARSE_FORMAT}"`);
    }
    validateOptionalNonNegativeInteger(errors, components.sparse_outliers, 'actual_nnz', `${label}.components.sparse_outliers`);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

export function assertFunctionalDescriptorManifest(value, label = 'descriptorManifest') {
  const validation = validateFunctionalDescriptorManifest(value, label);
  if (!validation.valid) {
    throw new Error(`Invalid ${label}:\n  - ${validation.errors.join('\n  - ')}`);
  }
  return value;
}
