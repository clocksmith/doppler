import { validateModelIR } from '../config/model-ir.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const TENSOR_ROLE_CLOSURE_SCHEMA_ID = 'doppler.tensor-role-closure/v1';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function digest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function requireObject(value, label) {
  if (!isObject(value)) throw new Error(`${label} must be an object.`);
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
  return value;
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer.`);
  return value;
}

function requireAuthor(author) {
  if (!isObject(author) || !['human', 'ai', 'tool'].includes(author.kind)
    || typeof author.actor !== 'string' || !author.actor.trim()) {
    throw new Error('Tensor-role closure requires attributable authorship.');
  }
}

function requireNode(nodes, predicate, label) {
  const node = nodes.find(predicate);
  if (!node) throw new Error(`ModelIR is missing ${label}.`);
  return node;
}

function sharedValue(nodes, section, field) {
  const values = nodes.map((node) => node[section]?.[field]);
  const first = values[0];
  if (values.some((value) => value !== first)) {
    throw new Error(`Tensor-role closure requires shared ${section}.${field} across scheduled block classes.`);
  }
  return requirePositiveInteger(first, `${section}.${field}`);
}

function buildDimensionSymbols(component, blockClasses) {
  return Object.freeze({
    hiddenSize: requirePositiveInteger(component.properties.hiddenSize, 'component hiddenSize'),
    vocabSize: requirePositiveInteger(component.properties.vocabSize, 'component vocabSize'),
    numLayers: requirePositiveInteger(component.properties.numLayers, 'component numLayers'),
    numHeads: sharedValue(blockClasses, 'geometry', 'numHeads'),
    numKvHeads: sharedValue(blockClasses, 'geometry', 'numKvHeads'),
    headDim: sharedValue(blockClasses, 'geometry', 'headDim'),
    intermediateSize: sharedValue(blockClasses, 'feedForward', 'intermediateSize'),
  });
}

function resolveDimension(expression, symbols, label) {
  if (Number.isInteger(expression) && expression > 0) return expression;
  const factors = requireString(expression, label).split('*');
  if (factors.some((factor) => !Object.hasOwn(symbols, factor))) {
    throw new Error(`${label} references an unsupported dimension expression "${expression}".`);
  }
  const value = factors.reduce((product, factor) => product * symbols[factor], 1);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} exceeds a safe positive integer.`);
  return value;
}

function resolveShape(shape, symbols, label) {
  if (!Array.isArray(shape) || shape.length === 0) throw new Error(`${label} must be a non-empty shape.`);
  return shape.map((dimension, index) => resolveDimension(dimension, symbols, `${label}[${index}]`));
}

function requireDescriptor(headers, tensorName, expectedDtype, expectedShape) {
  const descriptor = headers.tensors[tensorName];
  if (!descriptor) throw new Error(`Tensor-role closure is missing tensor "${tensorName}".`);
  if (descriptor.dtype !== expectedDtype) {
    throw new Error(`Tensor "${tensorName}" dtype mismatch: expected ${expectedDtype}, got ${descriptor.dtype}.`);
  }
  if (JSON.stringify(descriptor.shape) !== JSON.stringify(expectedShape)) {
    throw new Error(
      `Tensor "${tensorName}" shape mismatch: expected ${JSON.stringify(expectedShape)}, got ${JSON.stringify(descriptor.shape)}.`
    );
  }
  requireString(descriptor.sourceFile, `Tensor "${tensorName}" sourceFile`);
  return { tensorName, ...descriptor };
}

function bindingEvidence({ binding, names, headers, symbols, expectedDtype }) {
  const expectedShape = resolveShape(binding.shape, symbols, `binding "${binding.role}" shape`);
  const descriptors = names.map((name) => requireDescriptor(headers, name, expectedDtype, expectedShape));
  return {
    role: requireString(binding.role, 'tensor role'),
    expectedDtype,
    expectedShape,
    matchedTensors: descriptors.length,
    tensorEvidenceDigest: digest(descriptors),
    sourceFiles: [...new Set(descriptors.map((descriptor) => descriptor.sourceFile))].sort(),
  };
}

export function createTensorRoleClosureReceipt({ modelIR, headers, policy }) {
  const validation = validateModelIR(modelIR);
  if (!validation.ok || modelIR?.schema !== 'doppler.model-ir/v2') {
    throw new Error(`Tensor-role closure requires ModelIR v2: ${validation.errors.join('; ')}`);
  }
  requireObject(headers, 'SafeTensors header evidence');
  if (headers.schema !== 'doppler.safetensors-header-evidence/v1') {
    throw new Error('Tensor-role closure requires complete SafeTensors header evidence.');
  }
  requireObject(headers.tensors, 'SafeTensors tensors');
  requireObject(policy, 'Tensor-role closure policy');
  if (policy.schema !== TENSOR_ROLE_CLOSURE_SCHEMA_ID) {
    throw new Error(`Tensor-role closure requires policy "${TENSOR_ROLE_CLOSURE_SCHEMA_ID}".`);
  }
  requireAuthor(policy.author);
  const entryPoint = requireNode(
    modelIR.entryPoints,
    (candidate) => candidate.id === policy.entryPointId,
    `entry point "${policy.entryPointId}"`
  );
  if (entryPoint.status !== 'lowered') throw new Error(`Entry point "${entryPoint.id}" is not lowered.`);
  const component = requireNode(
    modelIR.components,
    (candidate) => candidate.id === entryPoint.componentId,
    `component "${entryPoint.componentId}"`
  );
  if (component.type !== policy.componentType) {
    throw new Error(`Entry-point component type "${component.type}" does not match policy "${policy.componentType}".`);
  }
  const schedule = requireNode(
    modelIR.blockSchedules,
    (candidate) => candidate.componentId === component.id,
    `block schedule for "${component.id}"`
  );
  if (schedule.blocks.length !== component.properties.numLayers) {
    throw new Error('Tensor-role closure requires a complete component block schedule.');
  }
  const classById = new Map(modelIR.blockClasses.map((blockClass) => [blockClass.id, blockClass]));
  const blockClasses = [...new Set(schedule.blocks.map((block) => block.blockClassId))]
    .map((blockClassId) => classById.get(blockClassId));
  if (blockClasses.some((blockClass) => !blockClass)) {
    throw new Error('Tensor-role closure schedule references an absent block class.');
  }
  const symbols = buildDimensionSymbols(component, blockClasses);
  const expectedDtype = requireString(policy.expectedDtype, 'expectedDtype');
  const expectedNames = new Set();
  const bindings = [];

  if (!Array.isArray(policy.rootBindings) || policy.rootBindings.length === 0) {
    throw new Error('Tensor-role closure requires rootBindings.');
  }
  for (const binding of policy.rootBindings) {
    requireObject(binding, 'root binding');
    const name = requireString(binding.name, `root binding "${binding.role}" name`);
    if (expectedNames.has(name)) throw new Error(`Tensor-role closure duplicates expected tensor "${name}".`);
    expectedNames.add(name);
    bindings.push({
      scope: 'component',
      ...bindingEvidence({ binding, names: [name], headers, symbols, expectedDtype }),
    });
  }

  if (!Array.isArray(policy.layerBindings) || policy.layerBindings.length === 0) {
    throw new Error('Tensor-role closure requires layerBindings.');
  }
  const layerPrefix = requireString(policy.layerPrefix, 'layerPrefix');
  if (!layerPrefix.includes('{L}')) throw new Error('layerPrefix must contain {L}.');
  for (const binding of policy.layerBindings) {
    requireObject(binding, 'layer binding');
    const suffix = requireString(binding.suffix, `layer binding "${binding.role}" suffix`);
    const names = schedule.blocks.map((block) => `${layerPrefix.replace('{L}', String(block.index))}.${suffix}`);
    for (const name of names) {
      if (expectedNames.has(name)) throw new Error(`Tensor-role closure duplicates expected tensor "${name}".`);
      expectedNames.add(name);
    }
    bindings.push({
      scope: 'layer',
      ...bindingEvidence({ binding, names, headers, symbols, expectedDtype }),
    });
  }

  if (!Array.isArray(policy.scopePrefixes) || policy.scopePrefixes.length === 0) {
    throw new Error('Tensor-role closure requires scopePrefixes.');
  }
  const scopePrefixes = policy.scopePrefixes.map((prefix, index) => requireString(prefix, `scopePrefixes[${index}]`));
  const rootNames = new Set(policy.rootBindings.map((binding) => binding.name));
  const observedNames = Object.keys(headers.tensors)
    .filter((name) => rootNames.has(name) || scopePrefixes.some((prefix) => name.startsWith(prefix)))
    .sort();
  const missingTensors = [...expectedNames].filter((name) => !Object.hasOwn(headers.tensors, name)).sort();
  const unexpectedTensors = observedNames.filter((name) => !expectedNames.has(name));
  if (missingTensors.length > 0 || unexpectedTensors.length > 0) {
    throw new Error(
      `Tensor-role closure failed: ${missingTensors.length} missing, ${unexpectedTensors.length} unexpected tensors.`
    );
  }

  const core = {
    schema: 'doppler.tensor-role-closure-receipt/v1',
    modelId: modelIR.modelId,
    entryPointId: entryPoint.id,
    sourceTopology: modelIR.supportScope.sourceTopology,
    author: structuredClone(policy.author),
    modelIRHash: digest(modelIR),
    headerEvidence: {
      schema: headers.schema,
      checkpointId: headers.checkpointId,
      repository: headers.repository,
      revision: headers.revision,
      sourceHeaderSha256: headers.sourceHeaderSha256,
      additionalSourceHeaders: structuredClone(headers.additionalSourceHeaders),
      tensorCount: headers.tensorCount,
      digest: digest(headers),
    },
    policyDigest: digest(policy),
    dimensionSymbols: symbols,
    bindings,
    expectedTensorCount: expectedNames.size,
    observedTensorCount: observedNames.length,
    outOfScopeTensorCount: Object.keys(headers.tensors).length - observedNames.length,
    missingTensors,
    unexpectedTensors,
    complete: true,
  };
  return Object.freeze({ ...core, receiptDigest: digest(core) });
}
