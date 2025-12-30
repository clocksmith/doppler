const REQUIRED_FIELDS = ['backward', 'grads'];

function assertObject(value, label) {
  if (!value || typeof value !== 'object') {
    throw new Error(`${label} must be an object`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertStringArray(value, label) {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new Error(`${label} must be an array of strings`);
  }
}

export function validateBackwardRegistry(registry) {
  assertObject(registry, 'Backward registry');
  assertObject(registry.ops, 'Backward registry ops');

  for (const [opName, entry] of Object.entries(registry.ops)) {
    assertObject(entry, `Backward registry op "${opName}"`);

    for (const field of REQUIRED_FIELDS) {
      if (!(field in entry)) {
        throw new Error(`Backward registry op "${opName}" is missing "${field}"`);
      }
    }

    assertString(entry.backward, `Backward registry op "${opName}".backward`);
    assertStringArray(entry.grads, `Backward registry op "${opName}".grads`);

    if (entry.requires_transpose != null && typeof entry.requires_transpose !== 'boolean') {
      throw new Error(`Backward registry op "${opName}".requires_transpose must be a boolean`);
    }
    if (entry.notes != null && typeof entry.notes !== 'string') {
      throw new Error(`Backward registry op "${opName}".notes must be a string`);
    }
  }

  return registry;
}
