import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { lowerChronosBoltGraph } from '../src/converter/chronos-bolt-graph.js';
import { createModelIRV2 } from '../src/config/model-ir-v2.js';
import { hashModelIR } from '../src/config/model-ir.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../src/formats/canonical-hash.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, i, values) => {
  if (value.startsWith('--')) pairs.push([value.slice(2), values[i + 1]]);
  return pairs;
}, []));
if (!args.source || !args.out) throw new Error('Usage: node tools/forge-chronos-bolt-pack.js --source DIR --out DIR');
const sourceRoot = path.resolve(args.source);
const outputRoot = path.resolve(args.out);
await fs.mkdir(outputRoot, { recursive: true });
const readJson = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const source = await readJson(path.join(sourceRoot, 'source-intake.json'));
const recipe = await readJson(path.join(root, 'src/config/forge/forecast/chronos-bolt-tiny.json'));
const registry = await readJson(path.join(root, 'src/config/kernels/registry.json'));
const artifacts = [];
async function artifact(artifactId, role, filename, bytes) {
  const destination = path.join(outputRoot, filename);
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(destination, bytes);
  const entry = { artifactId, role, path: filename, hash: hashBytesSha256(bytes), sizeBytes: bytes.byteLength };
  artifacts.push(entry);
  return entry;
}
for (const [filename, expected] of Object.entries(source.files)) {
  const bytes = await fs.readFile(path.join(sourceRoot, filename));
  if (hashBytesSha256(bytes) !== expected.hash || bytes.length !== expected.sizeBytes) throw new Error('Source custody mismatch: ' + filename);
  // Preserve one original checkpoint. Forge resolves its byte offsets; the
  // runtime never needs another copy or a SafeTensors parser.
  if (filename === 'weights.bin') continue;
  const role = filename === 'model.safetensors' ? 'weight-shard' : filename === 'reference.json' ? 'reference-report' : 'source-truth-evidence';
  await artifact(filename === 'model.safetensors' ? 'weights' : filename, role, 'source/' + filename, bytes);
}
const configBytes = await fs.readFile(path.join(sourceRoot, 'config.json'));
if (computeCanonicalSha256(JSON.parse(configBytes)) !== computeCanonicalSha256(source.config)) throw new Error('Source config facts drifted.');
const raw = await fs.readFile(path.join(sourceRoot, 'model.safetensors'));
const headerLength = Number(raw.readBigUInt64LE(0));
const header = JSON.parse(raw.subarray(8, 8 + headerLength));
const packed = await fs.readFile(path.join(sourceRoot, 'weights.bin'));
const executionSource = structuredClone(source);
for (const [name, entry] of Object.entries(source.tensors)) {
  const tensor = header[name];
  if (!tensor || JSON.stringify(tensor.shape) !== JSON.stringify(entry.shape) || tensor.dtype !== entry.dtype) throw new Error('Source tensor header mismatch: ' + name);
  const [begin, end] = tensor.data_offsets;
  const original = raw.subarray(8 + headerLength + begin, 8 + headerLength + end);
  const emitted = packed.subarray(entry.offsetBytes, entry.offsetBytes + entry.sizeBytes);
  if (hashBytesSha256(original) !== entry.hash || !original.equals(emitted)) throw new Error('Source tensor byte mismatch: ' + name);
  executionSource.tensors[name].offsetBytes = 8 + headerLength + begin;
}
const closureReceipt = { schema: 'doppler.chronos-source-closure/v1', source: computeCanonicalSha256(source),
  tensorCount: Object.keys(source.tensors).length, sourceBytesPreserved: true, configFieldsMatched: true };
await artifact('source-closure', 'source-truth-evidence', 'source/closure.json', new TextEncoder().encode(JSON.stringify(closureReceipt)));
const kernels = {};
const modules = [];
for (const [name, [operation, variant]] of Object.entries(recipe.kernels)) {
  const metadata = registry.operations[operation]?.variants[variant];
  if (!metadata || metadata.requires.length) throw new Error('Unresolved source-f32 kernel registry entry.');
  const bytes = await fs.readFile(path.join(root, 'src/gpu/kernels', metadata.wgsl));
  const normalized = bytes.toString('utf8').replace(/\r\n/g, '\n');
  const entry = metadata.entryPoint;
  const sourceArtifact = await artifact('wgsl-' + name, 'wgsl-source', 'kernels/' + metadata.wgsl, bytes);
  const module = { id: `${operation}-${variant}`, file: metadata.wgsl, entry, sourceArtifactId: sourceArtifact.artifactId,
    sourceHash: sourceArtifact.hash, digest: hashBytesSha256(new TextEncoder().encode(normalized + '\n@@entry:' + entry)) };
  kernels[name] = module;
  modules.push(module);
}
const graph = lowerChronosBoltGraph(executionSource, recipe, kernels);
await artifact('constants', 'weight-shard', 'constants.bin', graph.constantsBytes);
const execution = { steps: graph.steps };
const executionGraphHash = computeCanonicalSha256(execution);
const manifest = { schema: 'doppler.forecast-manifest/v1', modelId: recipe.modelId,
  forecast: graph.forecast, uploads: graph.uploads, executionGraphHash, execution };
const manifestArtifact = await artifact('manifest', 'manifest', 'manifest.json', new TextEncoder().encode(JSON.stringify(manifest)));
await artifact('lowering', 'conversion-config', 'lowering.json', new TextEncoder().encode(JSON.stringify(recipe)));
const facts = Object.entries(source.config).map(([field, value]) => ({ id: 'config-' + field, subject: recipe.modelId,
  predicate: field, value, confidence: 'direct', disposition: 'accepted', authorship: { kind: 'tool', actor: 'chronos-source-closure/v1' },
  validation: { status: 'passed', validator: 'chronos-source-closure/v1', receipt: computeCanonicalSha256(closureReceipt) },
  evidence: [{ kind: 'json-pointer', artifactId: 'config.json', file: 'config.json', pointer: '/' + field }] }));
for (const tensor of graph.tensorBindings) facts.push({ id: 'tensor-' + tensor.slotId, subject: tensor.name,
  predicate: 'source-tensor', value: { shape: tensor.shape, dtype: tensor.dtype, hash: tensor.hash },
  confidence: 'direct', disposition: 'accepted', authorship: { kind: 'tool', actor: 'chronos-source-closure/v1' },
  validation: { status: 'passed', validator: 'chronos-source-closure/v1', receipt: computeCanonicalSha256(closureReceipt) },
  evidence: [{ kind: 'tensor-header', artifactId: 'weights', file: 'model.safetensors', tensorName: tensor.name, shape: tensor.shape, dtype: tensor.dtype }] });
const refs = ['config-chronos_config', 'config-architectures', 'config-num_layers', 'config-num_decoder_layers'];
const modelIR = createModelIRV2({ modelId: recipe.modelId,
  sourceIdentity: { checkpointId: source.repository + '@' + source.revision, repository: source.repository, revision: source.revision,
    artifacts: artifacts.filter(a => a.role === 'source-truth-evidence' || a.artifactId === 'weights').map(a => ({ artifactId: a.artifactId, path: a.path, role: a.role, hash: a.hash })) },
  provenance: { forgeVersion: 'chronos-bolt-forge/v1', intakeDigest: computeCanonicalSha256(source), facts },
  components: [{ id: 'forecast', type: 't5-patch-encoder-decoder', role: 'numeric-quantile-forecast', properties: source.config, factRefs: refs }],
  blockClasses: ['encoder', 'decoder'].map(kind => ({ id: kind, kind, factRefs: refs,
    geometry: { hiddenSize: source.config.d_model, numHeads: source.config.num_heads, headDim: source.config.d_kv },
    normalization: { type: 'rms', epsilon: source.config.layer_norm_epsilon },
    positional: { type: 'relative-buckets', buckets: source.config.relative_attention_num_buckets, maxDistance: source.config.relative_attention_max_distance },
    feedForward: { activation: source.config.dense_act_fn, intermediateSize: source.config.d_ff, gated: false },
    phaseBehavior: { operation: 'forecast', cache: false } })),
  blockSchedules: ['encoder', 'decoder'].map(kind => ({ id: kind, componentId: 'forecast', factRefs: refs,
    blocks: Array.from({ length: kind === 'encoder' ? source.config.num_layers : source.config.num_decoder_layers }, (_, index) => ({ index, blockClassId: kind })) })),
  stateSpaces: [{ id: 'context', kind: 'numeric-context', persistence: 'request', contract: source.config.chronos_config, factRefs: refs }],
  tensorRoleBindings: graph.tensorBindings.map(t => ({ id: t.slotId, componentId: 'forecast', role: t.name,
    selector: { name: t.name, shape: t.shape, dtype: t.dtype }, factRefs: ['tensor-' + t.slotId] })),
  entryPoints: [{ id: 'forecast', componentId: 'forecast', kind: 'forecast', status: 'lowered', phases: ['forecast'], factRefs: refs }],
  outputHeads: [{ id: 'quantiles', componentId: 'forecast', kind: 'direct-quantile-forecast', factRefs: refs }],
  supportScope: { sourceTopology: 'complete', loweredEntryPoints: ['forecast'], qualifiedEntryPoints: [], unloweredEntryPoints: [] },
});
const bundle = { schema: 'doppler.forecast-program-bundle/v1', modelIRHash: hashModelIR(modelIR),
  executionGraphHash, manifestHash: manifestArtifact.hash, kernelClosure: modules, tensorBindings: graph.tensorBindings };
const bundleArtifact = await artifact('program', 'program-bundle', 'program.json', new TextEncoder().encode(JSON.stringify(bundle)));
const candidate = { modelId: recipe.modelId, modelIR, artifacts, wgslModules: modules,
  program: { schema: 'doppler.pack-program/v1', programBundleArtifactId: 'program', programBundleHash: bundleArtifact.hash,
    executionGraphHash, manifestArtifactId: 'manifest', modelIREvidenceArtifactId: 'source-closure',
    tokenizerArtifactIds: [], weightArtifactIds: ['weights', 'constants'], execution },
  targetPlan: { schema: 'doppler.target-plan/v1', schemaVersion: 1, targetId: 'webgpu-source-f32-context512',
    modelId: recipe.modelId, modelIRHash: hashModelIR(modelIR), executionGraphHash, programBundleHash: bundleArtifact.hash,
    capabilityPredicate: { requiresF16: false, requiresSubgroups: false, minBufferSize: Math.max(...graph.slots.map(s => s.size.bytes)) },
    dtypes: { activation: 'f32', weight: 'f32', kv: 'none' }, fusions: [],
    kernelClosure: modules.map(m => ({ moduleId: m.id, digest: m.digest, sourceHash: m.sourceHash })),
    memoryLayout: { kvCacheLayout: 'none', bufferSlots: graph.slots }, phases: { forecast: graph.steps }, qualification: [] } };
await fs.writeFile(path.join(outputRoot, 'candidate.json'), JSON.stringify(candidate));
console.log(JSON.stringify({ candidate: path.join(outputRoot, 'candidate.json'), commands: graph.steps.length,
  sourceTensors: graph.tensorBindings.length, gpuBytes: graph.slots.reduce((sum, s) => sum + s.size.bytes, 0),
  status: 'unqualified', executionGraphHash }));
