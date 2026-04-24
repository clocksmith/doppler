import catalog from './supported-operations.json' with { type: 'json' };

export const SUPPORTED_OPERATIONS_SCHEMA_ID = 'doppler.supported-operations/v1';

if (catalog.schema !== SUPPORTED_OPERATIONS_SCHEMA_ID) {
  throw new Error(
    `supported-operations.json schema mismatch: expected "${SUPPORTED_OPERATIONS_SCHEMA_ID}", got "${catalog.schema}".`
  );
}

export const SUPPORTED_EXECUTION_V1_OPS = Object.freeze(
  new Set(Object.keys(catalog.executionV1Ops))
);

export const SUPPORTED_RUNTIME_OPS = Object.freeze(
  new Set(catalog.runtimeOps.ops)
);

export function getSupportedOpsForFamily(family) {
  const entry = catalog.families?.[family];
  if (!entry || !Array.isArray(entry.requires)) {
    return null;
  }
  return Object.freeze(new Set(entry.requires));
}

export function listSupportedFamilies() {
  return Object.freeze(Object.keys(catalog.families ?? {}));
}

export function describeOp(op) {
  const entry = catalog.executionV1Ops?.[op];
  return entry ? Object.freeze({ ...entry }) : null;
}
