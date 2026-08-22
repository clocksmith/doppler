import fs from 'node:fs/promises';
import path from 'node:path';
import { parseSafetensorsHeader } from '../formats/safetensors/types.js';
import { computeCanonicalSha256 } from '../utils/canonical-hash.js';
import { sha256Hex } from '../utils/sha256.js';

export const SOURCE_INTAKE_SCHEMA = 'doppler.source-intake/v1';
export const SOURCE_INTAKE_CONVERSION_SKELETON_SCHEMA =
  'doppler.source-intake-conversion-skeleton/v1';
export const SOURCE_INTAKE_CONTRACT_TEST_SCHEMA =
  'doppler.source-intake-contract-test/v1';

const CONFIDENCE = new Set([
  'direct',
  'derived',
  'family-inferred',
  'ambiguous',
  'unsupported',
]);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function pointerValue(value, pointer) {
  const parts = pointer.split('/').slice(1).map((part) => (
    part.replaceAll('~1', '/').replaceAll('~0', '~')
  ));
  let current = value;
  for (const part of parts) {
    if (!isObject(current) && !Array.isArray(current)) return undefined;
    if (!Object.hasOwn(current, part)) return undefined;
    current = current[part];
  }
  return current;
}

function setOwnedValue(target, owner, value) {
  const parts = owner.split('.');
  let current = target;
  for (let index = 0; index < parts.length - 1; index += 1) {
    current[parts[index]] ??= {};
    current = current[parts[index]];
  }
  current[parts.at(-1)] = value;
}

function createFact({
  factId,
  file = null,
  jsonPointer = null,
  value = null,
  owner,
  proposal = value,
  confidence,
  status = confidence === 'direct' || confidence === 'derived' ? 'accepted' : 'unresolved',
  verificationBoundary = null,
  note = null,
}) {
  if (!CONFIDENCE.has(confidence)) {
    throw new Error(`source intake: invalid confidence "${confidence}" for ${factId}`);
  }
  return {
    factId,
    source: { file, jsonPointer, value },
    owner,
    proposal,
    confidence,
    status,
    verification: verificationBoundary
      ? { kind: 'boundary-pack', boundary: verificationBoundary }
      : { kind: 'contract-test', boundary: null },
    ...(note ? { note } : {}),
  };
}

async function readOptionalJson(sourceDir, filename) {
  const filePath = path.join(sourceDir, filename);
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    return {
      filename,
      raw,
      value: JSON.parse(raw),
      digest: `sha256:${sha256Hex(raw)}`,
    };
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw new Error(`source intake: unable to read ${filename}: ${error.message}`);
  }
}

async function readSafetensorsHeader(filePath) {
  const handle = await fs.open(filePath, 'r');
  try {
    const prefix = new Uint8Array(8);
    const prefixRead = await handle.read(prefix, 0, 8, 0);
    if (prefixRead.bytesRead !== 8) {
      throw new Error('header prefix is shorter than 8 bytes');
    }
    const headerSize = Number(new DataView(prefix.buffer).getBigUint64(0, true));
    const bytes = new Uint8Array(8 + headerSize);
    bytes.set(prefix);
    const headerRead = await handle.read(bytes, 8, headerSize, 8);
    if (headerRead.bytesRead !== headerSize) {
      throw new Error(`header is truncated (${headerRead.bytesRead}/${headerSize} bytes)`);
    }
    return parseSafetensorsHeader(bytes.buffer);
  } finally {
    await handle.close();
  }
}

function classifyTensor(name) {
  const rules = [
    ['token_embedding', /(embed_tokens|tok_embeddings|word_embeddings)\.weight$/],
    ['output_head', /(^|\.)(lm_head|output)\.weight$/],
    ['attention_q', /\.(q_proj|wq)\.weight$/],
    ['attention_k', /\.(k_proj|wk)\.weight$/],
    ['attention_v', /\.(v_proj|wv)\.weight$/],
    ['attention_output', /\.(o_proj|wo)\.weight$/],
    ['ffn_gate', /\.(gate_proj|w1)\.weight$/],
    ['ffn_up', /\.(up_proj|w3)\.weight$/],
    ['ffn_down', /\.(down_proj|w2)\.weight$/],
    ['normalization', /(norm|layernorm)\.weight$/],
  ];
  return rules.find(([, pattern]) => pattern.test(name))?.[0] ?? 'unknown';
}

function tensorLayerIndex(name) {
  const match = name.match(/(?:layers?|blocks?|h)\.(\d+)\./);
  return match ? Number(match[1]) : null;
}

async function collectTensorInventory(sourceDir) {
  const entries = await fs.readdir(sourceDir, { withFileTypes: true });
  const filenames = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.safetensors'))
    .map((entry) => entry.name)
    .sort();
  const tensors = [];
  const shards = [];
  for (const filename of filenames) {
    const parsed = await readSafetensorsHeader(path.join(sourceDir, filename));
    shards.push({
      file: filename,
      headerDigest: computeCanonicalSha256({
        metadata: parsed.metadata,
        tensors: parsed.tensors.map(({ name, dtype, shape, size }) => ({ name, dtype, shape, size })),
      }),
      tensorCount: parsed.tensors.length,
    });
    for (const tensor of parsed.tensors) {
      tensors.push({
        file: filename,
        name: tensor.name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        byteLength: tensor.size,
        role: classifyTensor(tensor.name),
        layerIndex: tensorLayerIndex(tensor.name),
      });
    }
  }
  return { shards, tensors };
}

function collectDirectFacts(policy, sources) {
  const facts = [];
  for (const entry of policy.facts) {
    let match = null;
    for (const source of sources) {
      for (const jsonPointer of entry.sourcePointers) {
        const value = pointerValue(source.value, jsonPointer);
        if (value !== undefined) {
          match = { source, jsonPointer, value };
          break;
        }
      }
      if (match) break;
    }
    if (!match) continue;
    facts.push(createFact({
      factId: entry.factId,
      file: match.source.filename,
      jsonPointer: match.jsonPointer,
      value: match.value,
      owner: entry.owner,
      confidence: 'direct',
      verificationBoundary: entry.verificationBoundary,
    }));
  }
  return facts;
}

function addDerivedFacts(facts) {
  const byId = new Map(facts.map((fact) => [fact.factId, fact]));
  if (!byId.has('architecture.attention.head_dim')) {
    const hiddenSize = byId.get('architecture.hidden_size')?.proposal;
    const queryHeads = byId.get('architecture.attention.query_heads')?.proposal;
    if (Number.isInteger(hiddenSize) && Number.isInteger(queryHeads) && hiddenSize % queryHeads === 0) {
      facts.push(createFact({
        factId: 'architecture.attention.head_dim',
        owner: 'manifest.architecture.headDim',
        proposal: hiddenSize / queryHeads,
        confidence: 'derived',
        note: 'Derived as hidden_size / num_attention_heads.',
      }));
    }
  }
}

function addRequiredFacts(policy, facts) {
  const existing = new Set(facts.map((fact) => fact.factId));
  for (const factId of policy.requiredFacts) {
    if (existing.has(factId)) continue;
    const entry = policy.facts.find((candidate) => candidate.factId === factId);
    facts.push(createFact({
      factId,
      owner: entry?.owner ?? 'unassigned',
      confidence: 'unsupported',
      note: 'Required source fact was not found in the inspected source files.',
    }));
  }
}

function addFamilyInferences(facts, familyIntake) {
  if (!familyIntake) return;
  if (familyIntake.schema !== SOURCE_INTAKE_SCHEMA) {
    throw new Error(`source intake: family intake must use ${SOURCE_INTAKE_SCHEMA}`);
  }
  const { digest, ...core } = familyIntake;
  if (computeCanonicalSha256(core) !== digest) {
    throw new Error('source intake: family intake digest does not match its payload');
  }
  const existing = new Set(facts.map((fact) => fact.factId));
  for (const familyFact of familyIntake.facts ?? []) {
    if (existing.has(familyFact.factId) || familyFact.status !== 'accepted') continue;
    facts.push(createFact({
      factId: familyFact.factId,
      file: familyFact.source?.file ?? null,
      jsonPointer: familyFact.source?.jsonPointer ?? null,
      value: familyFact.source?.value ?? null,
      owner: familyFact.owner,
      proposal: familyFact.proposal,
      confidence: 'family-inferred',
      verificationBoundary: familyFact.verification?.boundary ?? null,
      note: `Inherited only as a reviewable proposal from ${familyIntake.source?.digest ?? 'family intake'}.`,
    }));
  }
}

function addTensorFacts(facts, inventory) {
  const roles = {};
  for (const tensor of inventory.tensors) {
    roles[tensor.role] ??= [];
    roles[tensor.role].push(tensor);
  }
  const layerIndices = inventory.tensors
    .map((tensor) => tensor.layerIndex)
    .filter((value) => value !== null);
  const observedLayers = new Set(layerIndices).size;
  const configuredLayers = facts.find((fact) => fact.factId === 'architecture.layer_count')?.proposal;
  facts.push(createFact({
    factId: 'checkpoint.tensor_inventory',
    owner: 'converter.tensorMappings',
    proposal: {
      tensorCount: inventory.tensors.length,
      shardCount: inventory.shards.length,
      roles: Object.fromEntries(
        Object.entries(roles).map(([role, tensors]) => [role, tensors.length])
      ),
      observedLayerCount: observedLayers,
    },
    confidence: inventory.tensors.length > 0 ? 'direct' : 'unsupported',
    note: inventory.tensors.length > 0
      ? 'Derived directly from SafeTensors headers without reading tensor payloads.'
      : 'No SafeTensors files were found.',
  }));
  const unknownTensorNames = roles.unknown?.map((tensor) => tensor.name) ?? [];
  if (unknownTensorNames.length > 0) {
    facts.push(createFact({
      factId: 'checkpoint.unmapped_tensors',
      owner: 'converter.tensorMappings',
      proposal: unknownTensorNames,
      confidence: 'ambiguous',
      note: 'Checkpoint tensors without an explicit semantic role block source intake.',
    }));
  }
  if (Number.isInteger(configuredLayers) && observedLayers > 0 && configuredLayers !== observedLayers) {
    facts.push(createFact({
      factId: 'checkpoint.layer_pattern_mismatch',
      owner: 'converter.tensorMappings',
      proposal: { configuredLayers, observedLayers },
      confidence: 'ambiguous',
      note: 'Configured layer count does not match layer indices in SafeTensors headers.',
    }));
  }
  if (!roles.output_head?.length) {
    const tied = facts.find((fact) => fact.factId === 'architecture.tie_word_embeddings')?.proposal;
    facts.push(createFact({
      factId: 'checkpoint.output_head',
      owner: 'converter.tensorMappings.outputHead',
      proposal: tied === true ? 'tied-token-embedding' : null,
      confidence: tied === true ? 'derived' : 'ambiguous',
      verificationBoundary: 'model.logits',
      note: tied === true
        ? 'No output-head tensor exists; source config explicitly ties word embeddings.'
        : 'No output-head tensor exists and tied embeddings are not explicit.',
    }));
  }
}

function collectUnknownArchitectureFacts(policy, config, facts) {
  const claimedPointers = new Set(
    policy.facts.flatMap((entry) => entry.sourcePointers)
  );
  const architecturePattern = new RegExp(policy.architectureKeyPattern, 'i');
  const visit = (object, parentPointer = '') => {
    for (const [key, value] of Object.entries(object)) {
      const escapedKey = key.replaceAll('~', '~0').replaceAll('/', '~1');
      const pointer = `${parentPointer}/${escapedKey}`;
      if (claimedPointers.has(pointer)) continue;
      if (architecturePattern.test(key)) {
        facts.push(createFact({
          factId: `source.unmapped.${pointer.slice(1).replaceAll('/', '.')}`,
          file: config.filename,
          jsonPointer: pointer,
          value,
          owner: 'unassigned',
          confidence: 'ambiguous',
          note: 'Architecture-relevant source field has no Doppler intake owner mapping.',
        }));
        continue;
      }
      if (isObject(value)) visit(value, pointer);
    }
  };
  visit(config.value);
}

function buildProposedArtifacts(facts, sourceDigest) {
  const accepted = facts.filter((fact) => fact.status === 'accepted');
  const conversion = {
    schema: SOURCE_INTAKE_CONVERSION_SKELETON_SCHEMA,
    sourceIntakeDigest: sourceDigest,
    completeness: 'skeleton',
    acceptedFactCount: accepted.length,
    unresolvedFactIds: facts
      .filter((fact) => fact.status !== 'accepted')
      .map((fact) => fact.factId),
    proposal: {},
  };
  for (const fact of accepted) {
    if (fact.owner === 'unassigned' || fact.factId === 'checkpoint.tensor_inventory') continue;
    setOwnedValue(conversion.proposal, fact.owner, fact.proposal);
  }
  const contractTests = {
    schema: SOURCE_INTAKE_CONTRACT_TEST_SCHEMA,
    sourceIntakeDigest: sourceDigest,
    assertions: accepted.map((fact) => ({
      factId: fact.factId,
      owner: fact.owner,
      expected: fact.proposal,
      verification: fact.verification,
    })),
  };
  return { conversion, contractTests };
}

export async function inspectSourceModel({
  sourceDir,
  policy,
  familyIntake = null,
}) {
  if (!sourceDir) throw new Error('source intake: sourceDir is required');
  if (policy?.schema !== 'doppler.source-intake-policy/v1') {
    throw new Error('source intake: a doppler.source-intake-policy/v1 policy is required');
  }
  const resolvedSourceDir = path.resolve(sourceDir);
  const config = await readOptionalJson(resolvedSourceDir, 'config.json');
  if (!config) throw new Error(`source intake: config.json not found in ${resolvedSourceDir}`);
  const optionalSources = await Promise.all([
    readOptionalJson(resolvedSourceDir, 'generation_config.json'),
    readOptionalJson(resolvedSourceDir, 'tokenizer_config.json'),
  ]);
  const sources = [config, ...optionalSources.filter(Boolean)];
  const inventory = await collectTensorInventory(resolvedSourceDir);
  const facts = collectDirectFacts(policy, sources);
  addDerivedFacts(facts);
  addFamilyInferences(facts, familyIntake);
  addRequiredFacts(policy, facts);
  addTensorFacts(facts, inventory);
  collectUnknownArchitectureFacts(policy, config, facts);
  facts.sort((left, right) => left.factId.localeCompare(right.factId));

  const sourceIdentity = {
    directory: resolvedSourceDir,
    files: sources.map(({ filename, digest }) => ({ file: filename, digest })),
    safetensors: inventory.shards,
  };
  sourceIdentity.digest = computeCanonicalSha256(sourceIdentity);
  const unresolved = facts.filter((fact) => fact.status !== 'accepted');
  const reportCore = {
    schema: SOURCE_INTAKE_SCHEMA,
    ok: unresolved.every((fact) => (
      fact.confidence !== 'ambiguous' && fact.confidence !== 'unsupported'
    )),
    source: sourceIdentity,
    facts,
    summary: {
      accepted: facts.length - unresolved.length,
      unresolved: unresolved.length,
      blockers: unresolved.filter((fact) => (
        fact.confidence === 'ambiguous' || fact.confidence === 'unsupported'
      )).length,
      familyInferences: unresolved.filter((fact) => fact.confidence === 'family-inferred').length,
      unknownTensorNames: inventory.tensors
        .filter((tensor) => tensor.role === 'unknown')
        .map((tensor) => tensor.name),
    },
  };
  const report = {
    ...reportCore,
    digest: computeCanonicalSha256(reportCore),
  };
  return {
    report,
    artifacts: buildProposedArtifacts(facts, report.digest),
  };
}
