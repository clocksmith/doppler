#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MATRIX_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'local-gpu-challenger-matrix.json');
const SCHEMA_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'schema', 'local-gpu-challenger-matrix.schema.json');
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const REQUIRED_FAIRNESS_GATE_IDS = [
  'artifact-identity',
  'format-disclosure',
  'runtime-surface',
  'hardware-identity',
  'fallback-status',
  'cache-mode',
  'timing-scope',
  'correctness-first',
  'work-accounting',
  'sample-statistics',
  'claim-grade',
];

const ROLE_MODE = new Map([
  ['generation', 'text'],
  ['embedding', 'embedding'],
  ['rerank', 'rerank'],
]);

function asText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function toRepoPath(filePath) {
  return path.relative(REPO_ROOT, filePath).split(path.sep).join('/');
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function valueType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function resolveSchemaRef(ref, rootSchema, trace, errors) {
  if (!ref.startsWith('#/')) {
    errors.push(`${trace}: unsupported schema ref ${ref}`);
    return null;
  }
  const parts = ref.slice(2).split('/').map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'));
  let cursor = rootSchema;
  for (const part of parts) {
    if (!isObject(cursor) || !Object.prototype.hasOwnProperty.call(cursor, part)) {
      errors.push(`${trace}: unresolved schema ref ${ref}`);
      return null;
    }
    cursor = cursor[part];
  }
  return cursor;
}

function validateJsonSchemaValue(value, schema, rootSchema, trace, errors) {
  const label = trace || '$';
  if (!isObject(schema)) {
    errors.push(`${label}: schema must be an object`);
    return;
  }

  if (asText(schema.$ref)) {
    const referenced = resolveSchemaRef(schema.$ref, rootSchema, label, errors);
    if (!referenced) return;
    validateJsonSchemaValue(value, referenced, rootSchema, label, errors);
    const rest = Object.fromEntries(Object.entries(schema).filter(([key]) => key !== '$ref'));
    if (Object.keys(rest).length === 0) return;
    schema = rest;
  }

  if (schema.const !== undefined && value !== schema.const) {
    errors.push(`${label}: value must equal ${JSON.stringify(schema.const)}`);
  }
  if (Array.isArray(schema.enum) && !schema.enum.includes(value)) {
    errors.push(`${label}: value must be one of ${schema.enum.map((item) => JSON.stringify(item)).join(', ')}`);
  }

  if (schema.type !== undefined) {
    const expectedTypes = Array.isArray(schema.type) ? schema.type : [schema.type];
    const actualType = valueType(value);
    const integerMatch = expectedTypes.includes('integer') && typeof value === 'number' && Number.isInteger(value);
    if (!expectedTypes.includes(actualType) && !integerMatch) {
      errors.push(`${label}: expected ${expectedTypes.join(' | ')}, got ${actualType}`);
      return;
    }
    if (expectedTypes.includes('integer') && actualType === 'number' && !Number.isInteger(value)) {
      errors.push(`${label}: expected integer, got number`);
      return;
    }
  }

  if (typeof schema.minLength === 'number' && value !== null && value.length < schema.minLength) {
    errors.push(`${label}: minimum length is ${schema.minLength}`);
  }
  if (typeof schema.minItems === 'number' && Array.isArray(value) && value.length < schema.minItems) {
    errors.push(`${label}: minimum items is ${schema.minItems}`);
  }
  if (typeof schema.minProperties === 'number' && isObject(value) && Object.keys(value).length < schema.minProperties) {
    errors.push(`${label}: minimum properties is ${schema.minProperties}`);
  }
  if (typeof value === 'number') {
    if (typeof schema.minimum === 'number' && value < schema.minimum) {
      errors.push(`${label}: minimum is ${schema.minimum}`);
    }
    if (typeof schema.exclusiveMinimum === 'number' && value <= schema.exclusiveMinimum) {
      errors.push(`${label}: exclusive minimum is ${schema.exclusiveMinimum}`);
    }
    if (typeof schema.maximum === 'number' && value > schema.maximum) {
      errors.push(`${label}: maximum is ${schema.maximum}`);
    }
    if (typeof schema.exclusiveMaximum === 'number' && value >= schema.exclusiveMaximum) {
      errors.push(`${label}: exclusive maximum is ${schema.exclusiveMaximum}`);
    }
  }

  if (isObject(value) && Array.isArray(schema.required)) {
    for (const requiredKey of schema.required) {
      if (!Object.prototype.hasOwnProperty.call(value, requiredKey)) {
        errors.push(`${label}: required property "${requiredKey}" is missing`);
      }
    }
  }

  if (asText(schema.pattern) && typeof value === 'string') {
    const expression = new RegExp(schema.pattern);
    if (!expression.test(value)) {
      errors.push(`${label}: value does not match pattern ${schema.pattern}`);
    }
  }

  if (Array.isArray(value) && schema.items) {
    for (let i = 0; i < value.length; i += 1) {
      validateJsonSchemaValue(value[i], schema.items, rootSchema, `${label}[${i}]`, errors);
    }
  }

  if (isObject(value)) {
    if (isObject(schema.properties)) {
      for (const [key, nestedSchema] of Object.entries(schema.properties)) {
        if (Object.prototype.hasOwnProperty.call(value, key)) {
          validateJsonSchemaValue(value[key], nestedSchema, rootSchema, `${label}.${key}`, errors);
        }
      }
    }
    if (schema.additionalProperties === false) {
      const propertyKeys = new Set(isObject(schema.properties) ? Object.keys(schema.properties) : []);
      for (const key of Object.keys(value)) {
        if (!propertyKeys.has(key)) {
          errors.push(`${label}: unexpected property "${key}"`);
        }
      }
    }
  }
}

export function validateLocalGpuChallengerSchema(matrix, schema) {
  const errors = [];
  validateJsonSchemaValue(matrix, schema, schema, 'local GPU challenger matrix', errors);
  return errors;
}

function pushIf(condition, errors, message) {
  if (condition) errors.push(message);
}

function indexById(entries, label, errors) {
  const byId = new Map();
  if (!Array.isArray(entries)) {
    errors.push(`${label} must be an array`);
    return byId;
  }
  for (const entry of entries) {
    const id = asText(entry?.id);
    if (!id) {
      errors.push(`${label} entry is missing id`);
      continue;
    }
    if (byId.has(id)) {
      errors.push(`${label} has duplicate id: ${id}`);
      continue;
    }
    byId.set(id, entry);
  }
  return byId;
}

function buildCatalogByModelId(catalog) {
  return new Map(
    (Array.isArray(catalog?.models) ? catalog.models : [])
      .filter((entry) => asText(entry?.modelId))
      .map((entry) => [entry.modelId, entry])
  );
}

function normalizeModes(entry) {
  if (Array.isArray(entry?.modes)) return entry.modes.map(asText).filter(Boolean);
  const mode = asText(entry?.mode);
  return mode ? [mode] : [];
}

function validateAnchorComparator(model, catalogEntry, competitorById, anchorCompetitorId, errors) {
  const label = model.modelId;
  const anchor = model.anchorComparator;
  if (!isObject(anchor)) {
    errors.push(`${label}: anchorComparator must be an object`);
    return;
  }
  const anchorCompetitor = competitorById.get(anchor.competitorId);
  pushIf(!anchorCompetitor, errors, `${label}: unknown anchor competitor ${anchor.competitorId}`);
  pushIf(anchor.competitorId !== anchorCompetitorId, errors, `${label}: anchor competitor must be ${anchorCompetitorId}`);
  pushIf(anchorCompetitor?.claimScope !== 'anchor', errors, `${label}: anchor competitor must have claimScope anchor`);

  const tjs = catalogEntry?.vendorBenchmark?.transformersjs;
  if (isObject(tjs)) {
    pushIf(
      asText(tjs.repoId) && asText(tjs.repoId) !== anchor.modelId,
      errors,
      `${label}: anchor modelId must match catalog vendorBenchmark.transformersjs.repoId`
    );
    pushIf(
      asText(tjs.dtype) && asText(tjs.dtype) !== anchor.dtype,
      errors,
      `${label}: anchor dtype must match catalog vendorBenchmark.transformersjs.dtype`
    );
  }
}

function validateLocalChallengers(model, competitorById, minAdditionalLocalChallengers, errors) {
  const label = model.modelId;
  const challengers = Array.isArray(model.localChallengers) ? model.localChallengers : [];
  pushIf(
    challengers.length < minAdditionalLocalChallengers,
    errors,
    `${label}: must list at least ${minAdditionalLocalChallengers} local challengers`
  );
  const seenCompetitors = new Set();
  for (const challenger of challengers) {
    const competitorId = asText(challenger?.competitorId);
    const competitor = competitorById.get(competitorId);
    pushIf(!competitor, errors, `${label}: unknown local challenger ${competitorId}`);
    pushIf(competitor?.claimScope !== 'local-challenger', errors, `${label}: ${competitorId} must be a local challenger`);
    pushIf(seenCompetitors.has(competitorId), errors, `${label}: duplicate local challenger ${competitorId}`);
    seenCompetitors.add(competitorId);
    pushIf(!asText(challenger?.status), errors, `${label}: ${competitorId} must declare status`);
    pushIf(!asText(challenger?.nextGate), errors, `${label}: ${competitorId} must declare nextGate`);
  }
}

function validateHarnesses(harnessById, fairnessGateById, claimGradeOrder, errors) {
  for (const [harnessId, harness] of harnessById) {
    const gates = Array.isArray(harness.fairnessGates) ? harness.fairnessGates : [];
    for (const gateId of REQUIRED_FAIRNESS_GATE_IDS) {
      pushIf(!gates.includes(gateId), errors, `${harnessId}: missing fairness gate ${gateId}`);
    }
    for (const gateId of gates) {
      pushIf(!fairnessGateById.has(gateId), errors, `${harnessId}: unknown fairness gate ${gateId}`);
    }
    pushIf(
      JSON.stringify(harness.claimGrades) !== JSON.stringify(claimGradeOrder),
      errors,
      `${harnessId}: claimGrades must match selectionPolicy.claimGradeOrder`
    );
    const overlay = harness.engineOverlayPolicy;
    pushIf(!isObject(overlay), errors, `${harnessId}: engineOverlayPolicy must be an object`);
    pushIf(
      !Array.isArray(overlay?.forbiddenSharedFields) || overlay.forbiddenSharedFields.length === 0,
      errors,
      `${harnessId}: forbiddenSharedFields must be explicit`
    );
  }
}

function validatePlatformTargets(matrix, errors) {
  const platformTargets = Array.isArray(matrix.platformTargets) ? matrix.platformTargets : [];
  pushIf(platformTargets.length === 0, errors, 'platformTargets must be a non-empty array');
  const platformById = indexById(platformTargets, 'platformTargets', errors);
  pushIf(
    !platformTargets.some((target) => target.status === 'local-probe-supported'),
    errors,
    'platformTargets must identify at least one local-probe-supported target'
  );
  pushIf(!platformById.has('apple-metal'), errors, 'platformTargets must leave Apple Metal open');
  pushIf(!platformById.has('linux-amd-vulkan-rocm'), errors, 'platformTargets must include Linux AMD Vulkan/ROCm');
  pushIf(!platformById.has('linux-nvidia-vulkan-cuda'), errors, 'platformTargets must leave Linux NVIDIA open');
  pushIf(!platformById.has('windows-nvidia-webgpu-cuda'), errors, 'platformTargets must leave Windows NVIDIA open');
  pushIf(!platformById.has('windows-amd-webgpu-directml'), errors, 'platformTargets must leave Windows AMD open');
  pushIf(!platformById.has('windows-intel-webgpu'), errors, 'platformTargets must leave Windows Intel open');
  pushIf(!platformById.has('nvidia-orin-spark-linux'), errors, 'platformTargets must leave NVIDIA Orin/Spark open');
}

export function validateLocalGpuChallengerMatrix(matrix, catalog, schema = null) {
  const errors = [];
  if (!isObject(matrix)) {
    return {
      ok: false,
      errors: ['local GPU challenger matrix must be an object'],
    };
  }

  if (schema !== null) {
    errors.push(...validateLocalGpuChallengerSchema(matrix, schema));
  }

  pushIf(matrix.schemaVersion !== 1, errors, 'schemaVersion must be 1');
  pushIf(matrix.matrixId !== 'local-gpu-challenger-matrix', errors, 'matrixId must be local-gpu-challenger-matrix');
  pushIf(matrix.$schema !== 'schema/local-gpu-challenger-matrix.schema.json', errors, '$schema must point at the local schema');

  const selectionPolicy = isObject(matrix.selectionPolicy) ? matrix.selectionPolicy : {};
  const anchorCompetitorId = asText(selectionPolicy.anchorCompetitorId);
  const minAdditionalLocalChallengers = Number(selectionPolicy.minAdditionalLocalChallengers);
  const claimGradeOrder = Array.isArray(selectionPolicy.claimGradeOrder) ? selectionPolicy.claimGradeOrder : [];
  const fairnessGateById = indexById(matrix.fairnessGates, 'fairnessGates', errors);
  const competitorById = indexById(matrix.competitors, 'competitors', errors);
  const harnessById = indexById(matrix.harnesses, 'harnesses', errors);
  const catalogByModelId = buildCatalogByModelId(catalog);

  pushIf(!competitorById.has(anchorCompetitorId), errors, `anchor competitor is unknown: ${anchorCompetitorId}`);
  pushIf(selectionPolicy.hostClass !== 'multi-platform-local-gpu', errors, 'selectionPolicy.hostClass must be multi-platform-local-gpu');
  pushIf(!asText(selectionPolicy.probeHostClass), errors, 'selectionPolicy.probeHostClass is required');
  validatePlatformTargets(matrix, errors);
  for (const gateId of REQUIRED_FAIRNESS_GATE_IDS) {
    pushIf(!fairnessGateById.has(gateId), errors, `fairnessGates must include ${gateId}`);
  }
  for (const [competitorId, competitor] of competitorById) {
    pushIf(competitor.fallbackPolicy !== 'fail-closed', errors, `${competitorId}: fallbackPolicy must be fail-closed`);
    pushIf(!asText(competitor.evidenceKind), errors, `${competitorId}: evidenceKind is required`);
  }

  validateHarnesses(harnessById, fairnessGateById, claimGradeOrder, errors);

  const seenModelIds = new Set();
  const models = Array.isArray(matrix.models) ? matrix.models : [];
  pushIf(models.length === 0, errors, 'models must be a non-empty array');
  for (const model of models) {
    const modelId = asText(model?.modelId);
    if (!modelId) {
      errors.push('model entry is missing modelId');
      continue;
    }
    pushIf(seenModelIds.has(modelId), errors, `duplicate modelId: ${modelId}`);
    seenModelIds.add(modelId);

    const catalogEntry = catalogByModelId.get(modelId);
    pushIf(!catalogEntry, errors, `${modelId}: must exist in models/catalog.json`);
    const expectedMode = ROLE_MODE.get(model.modelRole);
    const catalogModes = normalizeModes(catalogEntry);
    pushIf(
      expectedMode && catalogEntry && !catalogModes.includes(expectedMode),
      errors,
      `${modelId}: modelRole ${model.modelRole} requires catalog mode ${expectedMode}`
    );
    pushIf(model.dopplerArtifactId !== modelId, errors, `${modelId}: dopplerArtifactId must match modelId for the selected artifact`);
    for (const alternateId of model.alternateDopplerArtifactIds || []) {
      pushIf(!catalogByModelId.has(alternateId), errors, `${modelId}: alternate artifact is missing from catalog: ${alternateId}`);
    }

    validateAnchorComparator(model, catalogEntry, competitorById, anchorCompetitorId, errors);
    validateLocalChallengers(model, competitorById, minAdditionalLocalChallengers, errors);

    const recommendedHarnesses = Array.isArray(model.recommendedHarnesses) ? model.recommendedHarnesses : [];
    pushIf(recommendedHarnesses.length === 0, errors, `${modelId}: recommendedHarnesses must be non-empty`);
    for (const harnessId of recommendedHarnesses) {
      pushIf(!harnessById.has(harnessId), errors, `${modelId}: unknown recommended harness ${harnessId}`);
    }
    pushIf(
      !claimGradeOrder.includes(model.claimPolicy?.minimumClaimGrade),
      errors,
      `${modelId}: claimPolicy.minimumClaimGrade must be in selectionPolicy.claimGradeOrder`
    );
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}

export function buildLocalGpuChallengerReport(matrix, catalog, options = {}) {
  const validation = validateLocalGpuChallengerMatrix(matrix, catalog, options.schema || null);
  const competitorById = indexById(matrix.competitors || [], 'competitors', []);
  const rows = (Array.isArray(matrix.models) ? matrix.models : []).map((model) => {
    const challengers = (Array.isArray(model.localChallengers) ? model.localChallengers : []).map((challenger) => ({
      competitorId: challenger.competitorId,
      label: competitorById.get(challenger.competitorId)?.label || challenger.competitorId,
      status: challenger.status,
      nextGate: challenger.nextGate,
      candidateArtifact: challenger.candidateArtifact,
      latestEvidence: challenger.latestEvidence || null,
    }));
    return {
      modelId: model.modelId,
      publicModelName: model.publicModelName,
      tier: model.tier,
      modelRole: model.modelRole,
      sourceCheckpoint: model.sourceCheckpoint,
      dopplerArtifactId: model.dopplerArtifactId,
      alternateDopplerArtifactIds: model.alternateDopplerArtifactIds,
      anchorComparator: model.anchorComparator,
      latestAnchorEvidence: model.latestAnchorEvidence || null,
      localChallengers: challengers,
      recommendedHarnesses: model.recommendedHarnesses,
      minimumClaimGrade: model.claimPolicy?.minimumClaimGrade,
    };
  });
  const gaps = rows.flatMap((row) => row.localChallengers
    .filter((challenger) => challenger.status !== 'configured')
    .map((challenger) => ({
      modelId: row.modelId,
      publicModelName: row.publicModelName,
      competitorId: challenger.competitorId,
      status: challenger.status,
      nextGate: challenger.nextGate,
    })));

  const tierCounts = {};
  for (const row of rows) {
    tierCounts[row.tier] = (tierCounts[row.tier] || 0) + 1;
  }

  const report = {
    ok: validation.ok,
    matrixPath: toRepoPath(MATRIX_PATH),
    schemaPath: toRepoPath(SCHEMA_PATH),
    catalogPath: toRepoPath(CATALOG_PATH),
    schemaVersion: matrix.schemaVersion,
    updated: matrix.updated,
    hostClass: matrix.selectionPolicy?.hostClass,
    probeHostClass: matrix.selectionPolicy?.probeHostClass,
    summary: {
      models: rows.length,
      competitors: Array.isArray(matrix.competitors) ? matrix.competitors.length : 0,
      harnesses: Array.isArray(matrix.harnesses) ? matrix.harnesses.length : 0,
      fairnessGates: Array.isArray(matrix.fairnessGates) ? matrix.fairnessGates.length : 0,
      platformTargets: Array.isArray(matrix.platformTargets) ? matrix.platformTargets.length : 0,
      gaps: gaps.length,
      tierCounts,
    },
    platformTargets: Array.isArray(matrix.platformTargets) ? matrix.platformTargets : [],
    rows,
    gaps,
    errors: validation.errors,
  };

  if (options.probeLocal === true) {
    report.localProbe = probeLocalHost();
  }

  return report;
}

function probeCommand(command, args = []) {
  const result = spawnSync(command, args, {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return {
    command: [command, ...args].join(' '),
    available: !result.error && result.status === 0,
    status: result.status,
    output: `${result.stdout || ''}${result.stderr || ''}`.trim().slice(0, 4000),
    error: result.error?.code || null,
  };
}

function probeNodeImport(specifier) {
  const script = `await import(${JSON.stringify(specifier)}); console.log('ok')`;
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', script], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return {
    specifier,
    available: !result.error && result.status === 0,
    status: result.status,
    output: `${result.stdout || ''}${result.stderr || ''}`.trim().slice(0, 4000),
    error: result.error?.code || null,
  };
}

function probePythonTorch() {
  const script = [
    'import json',
    'try:',
    '    import torch',
    '    cuda_available = bool(torch.cuda.is_available())',
    '    hip_version = getattr(torch.version, "hip", None)',
    '    cuda_version = getattr(torch.version, "cuda", None)',
    '    if hip_version:',
    '        backend = "rocm" if cuda_available else "rocm-build-no-device"',
    '    elif cuda_version:',
    '        backend = "cuda" if cuda_available else "cuda-build-no-device"',
    '    else:',
    '        backend = "cpu-only"',
    '    print(json.dumps({',
    '        "available": True,',
    '        "version": getattr(torch, "__version__", None),',
    '        "cudaAvailable": cuda_available,',
    '        "cudaVersion": cuda_version,',
    '        "hipVersion": hip_version,',
    '        "backend": backend,',
    '        "gpuUsable": bool(cuda_available and (hip_version or cuda_version)),',
    '        "importError": None',
    '    }, sort_keys=True))',
    'except Exception as exc:',
    '    print(json.dumps({',
    '        "available": False,',
    '        "version": None,',
    '        "cudaAvailable": False,',
    '        "cudaVersion": None,',
    '        "hipVersion": None,',
    '        "backend": "unavailable",',
    '        "gpuUsable": False,',
    '        "importError": f"{type(exc).__name__}: {exc}"',
    '    }, sort_keys=True))',
  ].join('\n');
  const result = spawnSync('python3', ['-c', script], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const output = `${result.stdout || ''}${result.stderr || ''}`.trim().slice(0, 4000);
  let details = null;
  try {
    details = JSON.parse((result.stdout || '').trim());
  } catch {
    details = null;
  }
  return {
    command: 'python3 -c <torch probe>',
    available: !result.error && result.status === 0 && details?.available === true,
    status: result.status,
    output,
    error: result.error?.code || null,
    version: details?.version || null,
    cudaAvailable: details?.cudaAvailable === true,
    cudaVersion: details?.cudaVersion || null,
    hipVersion: details?.hipVersion || null,
    backend: details?.backend || 'unavailable',
    gpuUsable: details?.gpuUsable === true,
    importError: details?.importError || null,
  };
}

export function probeLocalHost() {
  return {
    commands: {
      node: probeCommand(process.execPath, ['--version']),
      vulkaninfo: probeCommand('vulkaninfo', ['--summary']),
      rocminfo: probeCommand('rocminfo'),
      rocmSmi: probeCommand('rocm-smi'),
      llamaBench: probeCommand('llama-bench', ['--help']),
      llamaCli: probeCommand('llama-cli', ['--help']),
      ollama: probeCommand('ollama', ['--version']),
    },
    nodePackages: {
      transformersjs: probeNodeImport('@huggingface/transformers'),
      onnxruntimeWeb: probeNodeImport('onnxruntime-web'),
      onnxruntimeNode: probeNodeImport('onnxruntime-node'),
    },
    python: {
      torch: probePythonTorch(),
    },
  };
}

function formatOptional(value) {
  return value == null || value === '' ? 'none' : String(value);
}

function formatReport(report) {
  const lines = [
    `Local GPU Challenger Matrix (${report.updated})`,
    `Models: ${report.summary.models}; competitors: ${report.summary.competitors}; harnesses: ${report.summary.harnesses}; platform targets: ${report.summary.platformTargets}; open challenger gates: ${report.summary.gaps}`,
    `Host class: ${report.hostClass}; probe host: ${report.probeHostClass}`,
    '',
  ];
  for (const row of report.rows) {
    const challengers = row.localChallengers
      .map((entry) => `${entry.competitorId}:${entry.status}`)
      .join(', ');
    lines.push(`${row.tier} | ${row.publicModelName} | ${row.modelRole}`);
    lines.push(`  modelId: ${row.modelId}`);
    lines.push(`  anchor: ${row.anchorComparator.competitorId} -> ${row.anchorComparator.modelId} (${row.anchorComparator.status})`);
    if (row.latestAnchorEvidence) {
      lines.push(`  latest evidence: ${row.latestAnchorEvidence.winner} / ${row.latestAnchorEvidence.claimGrade} / ${row.latestAnchorEvidence.receiptPath}`);
    }
    for (const challenger of row.localChallengers) {
      if (!challenger.latestEvidence) continue;
      const evidence = challenger.latestEvidence;
      lines.push(`  local evidence (${challenger.competitorId}): ${evidence.winner} / ${evidence.claimGrade} / ${evidence.receiptPaths.join(', ')}`);
    }
    lines.push(`  challengers: ${challengers}`);
    lines.push(`  harnesses: ${row.recommendedHarnesses.join(', ')}`);
  }
  if (report.errors.length > 0) {
    lines.push('');
    lines.push('Errors:');
    for (const error of report.errors) {
      lines.push(`- ${error}`);
    }
  }
  if (report.localProbe) {
    lines.push('');
    lines.push('Local Probe:');
    for (const [id, probe] of Object.entries(report.localProbe.commands)) {
      lines.push(`- ${id}: ${probe.available ? 'available' : 'unavailable'}`);
    }
    for (const [id, probe] of Object.entries(report.localProbe.nodePackages)) {
      lines.push(`- ${id}: ${probe.available ? 'available' : 'unavailable'}`);
    }
    if (report.localProbe.python?.torch) {
      const torch = report.localProbe.python.torch;
      lines.push([
        `- python.torch: ${torch.available ? 'available' : 'unavailable'}`,
        `backend=${formatOptional(torch.backend)}`,
        `gpuUsable=${torch.gpuUsable === true ? 'true' : 'false'}`,
        `cudaAvailable=${torch.cudaAvailable === true ? 'true' : 'false'}`,
        `cuda=${formatOptional(torch.cudaVersion)}`,
        `hip=${formatOptional(torch.hipVersion)}`,
      ].join('; '));
    }
  }
  return lines.join('\n');
}

function parseArgs(argv) {
  const flags = {
    check: false,
    json: false,
    probeLocal: false,
  };
  for (const token of argv) {
    if (token === '--check') {
      flags.check = true;
    } else if (token === '--json') {
      flags.json = true;
    } else if (token === '--probe-local') {
      flags.probeLocal = true;
    } else if (token === '--help' || token === '-h') {
      flags.help = true;
    } else {
      throw new Error(`Unknown argument: ${token}`);
    }
  }
  return flags;
}

function usage() {
  return [
    'Usage:',
    '  node tools/local-gpu-challengers.js [--json] [--probe-local]',
    '  node tools/local-gpu-challengers.js --check',
  ].join('\n');
}

export async function main(argv = process.argv.slice(2)) {
  const flags = parseArgs(argv);
  if (flags.help) {
    console.log(usage());
    return;
  }
  const [matrix, catalog, schema] = await Promise.all([
    readJson(MATRIX_PATH),
    readJson(CATALOG_PATH),
    readJson(SCHEMA_PATH),
  ]);
  const report = buildLocalGpuChallengerReport(matrix, catalog, {
    schema,
    probeLocal: flags.probeLocal,
  });

  if (flags.json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (flags.check && report.ok) {
    console.log(`local-gpu-challengers: ok (${report.summary.models} models, ${report.summary.gaps} open challenger gates)`);
  } else {
    console.log(formatReport(report));
  }

  if (!report.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
