// Model names: lowercase letters, digits and single hyphens (docs/style/model-naming.md).
// A manifest carries its model ID, so a badly formed name becomes part of an identity. These checks keep new
// names well formed and keep the generated suffix honest about what it describes.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const {
  buildModelNameSuffix,
  buildQuantizationInfo,
  buildVariantTag,
  normalizeQuantTag,
  quantNameToken,
  resolveModelId,
} = await import('../../src/converter/quantization-info.js');
const {
  assertCanonicalModelId,
  assertModelIdDescribesFormat,
  formatRecordProblems,
  isCanonicalModelId,
  isLegacyModelId,
  LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS,
  LEGACY_NONCANONICAL_MODEL_IDS,
  modelIdSuffixProblem,
  modelLabelFromFileName,
  variantNameSuffix,
} = await import('../../src/converter/model-naming.js');
const { resolveConvertedModelId } = await import('../../src/converter/conversion-plan.js');

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

// The plain GGUF block formats are spelled with an underscore in manifests and without one in names.
assert.equal(quantNameToken('q4_0'), 'q40');
assert.equal(quantNameToken('Q8_0'), 'q80');
assert.equal(quantNameToken('q4_1'), 'q41');
assert.equal(quantNameToken('q5_0'), 'q50');
assert.equal(quantNameToken('Q4_K_M'), 'q4k');
assert.equal(quantNameToken('f16'), 'f16');
// A name token is accepted wherever a quantization is written, and means the same format.
assert.equal(normalizeQuantTag('q40'), 'q4_0');
assert.equal(normalizeQuantTag('q80'), 'q8_0');

// The tag a manifest stores names the weight format, then every part stored or computed differently from it.
// It keeps the GGUF spelling: rewriting it would change the identity of every model converted again.
assert.equal(buildVariantTag({ weights: 'q4k', embeddings: 'f16', lmHead: 'f16', compute: 'f32', layout: 'row' }), 'q4k-ehf16-af32');
assert.equal(buildVariantTag({ weights: 'q4k', embeddings: 'f16', lmHead: 'q4k', compute: 'f32', layout: 'row' }), 'q4k-ef16-af32');
assert.equal(buildVariantTag({ weights: 'q4_0', embeddings: 'q4_0', lmHead: 'q8_0', compute: 'f32' }), 'q4_0-hq8_0-af32');
assert.equal(buildVariantTag({ weights: 'q4_0', embeddings: 'q4_0', lmHead: 'q6k', compute: 'f32', experts: 'q4_0' }), 'q4_0-hq6k-af32');
// In a name the same tag has no underscore.
assert.equal(variantNameSuffix('q4_0-hq8_0-af32'), 'q40-hq80-af32');
assert.equal(variantNameSuffix('q4_0-hq6k-af32'), 'q40-hq6k-af32');
assert.equal(variantNameSuffix('q4k-ehf16-af32'), 'q4k-ehf16-af32');
assert.equal(variantNameSuffix('f16-af32-xq8_0'), 'f16-af32-xq80');
assert.equal(variantNameSuffix(null), '');
for (const tag of ['q4_0-hq8_0-af32', 'q4k-ehf16-af32', 'f16-af32-xq8_0']) assert.ok(isCanonicalModelId(`model-${variantNameSuffix(tag)}`), tag);

// A tag written before the rule is brought into line when it is joined to a name, and is not appended twice.
assert.equal(resolveModelId('Qwen-3.6-35B-A3B-text-source', 'unused', 'q4_0-hq6k-af32'), 'qwen-3-6-35b-a3b-text-source-q40-hq6k-af32');
assert.equal(resolveModelId('qwen-3-5-2b-q4k-ef16-af32', 'unused', 'q4k-ef16-af32'), 'qwen-3-5-2b-q4k-ef16-af32');

// The manifest retains its original tag. Names distinguish activation precision even when weights share it.
for (const [info, stored, suffix] of [
  [{ weights: 'q4k', embeddings: 'f16', lmHead: 'f16', compute: 'f16' }, 'q4k-ehaf16', 'q4k-ehf16-af16'],
  [{ weights: 'q4k', embeddings: 'f16', lmHead: 'q6k', compute: 'f16', layout: 'col' }, 'q4k-col-eaf16-hq6k', 'q4k-col-ef16-hq6k-af16'],
  [{ weights: 'q4k', embeddings: 'q4k', lmHead: 'f16', compute: 'f16', audio: 'f16' }, 'q4k-haf16-audiof16', 'q4k-hf16-af16-audiof16'],
  [{ weights: 'f32', compute: 'f32' }, 'f32', 'f32-af32'],
  [{ weights: 'f16', compute: 'f16', experts: 'mxfp4' }, 'f16-xmxfp4', 'f16-af16-xmxfp4'],
  [{ weights: 'f32' }, 'f32', 'f32'],
]) {
  assert.equal(buildVariantTag(info), stored);
  assert.equal(buildModelNameSuffix(info), suffix);
  const actual = resolveConvertedModelId({ detectedModelId: 'test-model', quantizationInfo: { ...info, variantTag: stored } });
  assert.equal(actual, `test-model-${suffix}`);
  assert.equal(resolveConvertedModelId({ detectedModelId: `test-model-${stored}`, quantizationInfo: { ...info, variantTag: stored } }), actual);
}
assert.equal(variantNameSuffix('q4k-eaf16-hq6k-vf16'), 'q4k-ef16-hq6k-af16-vf16');
assert.throws(() => resolveConvertedModelId({ explicitModelId: 'new-model-q4k-ehaf16', quantizationInfo: { variantTag: 'q4k-ehaf16' } }), /does not describe/);
assert.equal(resolveConvertedModelId({ explicitModelId: 'qwen-3-6-27b-q4k-eaf16', quantizationInfo: { variantTag: 'q4k-eaf16' } }), 'qwen-3-6-27b-q4k-eaf16');

for (const good of ['qwen-3-5-2b-q4k-ehf16-af32', 'gemma-3-270m-it-f16-af32', 'a1']) assert.ok(isCanonicalModelId(good), good);
for (const bad of ['qwen_3', 'Qwen-3', 'qwen--3', '-qwen', 'qwen-', 'qwen 3', 'qwen.3', '', null, 42]) assert.ok(!isCanonicalModelId(bad), String(bad));

// A new nonconforming ID is refused before it can enter a manifest; an identity issued earlier still converts.
assert.throws(() => assertCanonicalModelId('new-model-q4_0-af32'), /lowercase letters, digits and single hyphens/);
assert.throws(() => resolveConvertedModelId({ converterConfig: { output: { modelBaseId: 'new-model-q4_0-af32' } } }), /not a valid model name/);
for (const legacy of LEGACY_NONCANONICAL_MODEL_IDS) {
  assert.equal(resolveConvertedModelId({ converterConfig: { output: { modelBaseId: legacy } } }), legacy);
}
assert.equal(resolveConvertedModelId({ detectedModelId: 'Some_Model', quantizationInfo: { variantTag: 'q4_0-af32' } }), 'some-model-q40-af32');

// The production converter passes the configured name as the explicit ID. That path is checked too.
assert.throws(() => resolveConvertedModelId({ explicitModelId: 'new_model' }), /not a valid model name/);
assert.throws(() => resolveConvertedModelId({ explicitModelId: 'New.Model_1' }), /not a valid model name/);
assert.equal(resolveConvertedModelId({ explicitModelId: LEGACY_NONCANONICAL_MODEL_IDS[0] }), LEGACY_NONCANONICAL_MODEL_IDS[0]);

// A conversion's name must end with the suffix its format generates, after at least one other part.
const conversion = variantTag => ({ quantizationInfo: { variantTag } });
assert.equal(modelIdSuffixProblem('qwen-3-5-2b-q4k-ef16-af32', 'q4k-ef16-af32'), null);
assert.equal(modelIdSuffixProblem('qwen-3-6-35b-a3b-text-source-q40-hq6k-af32', 'q4_0-hq6k-af32'), null);
assert.match(modelIdSuffixProblem('qwen-3-5-2b-q4k-ehaf16', 'q4k-ef16-af32'), /must end with "-q4k-ef16-af32"/);
assert.match(modelIdSuffixProblem('q4k-ef16-af32', 'q4k-ef16-af32'), /must end with/);
assert.match(modelIdSuffixProblem('some-model', null), /format is unknown/);
assert.equal(resolveConvertedModelId({ explicitModelId: 'qwen-3-5-2b-q4k-ef16-af32', ...conversion('q4k-ef16-af32') }), 'qwen-3-5-2b-q4k-ef16-af32');
assert.throws(() => resolveConvertedModelId({ explicitModelId: 'a1', ...conversion('q4k-ef16-af32') }), /does not describe its model: it must end with "-q4k-ef16-af32"/);
assert.throws(() => resolveConvertedModelId({ explicitModelId: 'new-2b-q4k-ehaf16', ...conversion('q4k-ef16-af32') }), /does not describe its model/);
assert.throws(() => resolveConvertedModelId({ converterConfig: { output: { modelBaseId: 'new-2b-q4k-ehaf16' } }, ...conversion('q4k-ef16-af32') }), /does not describe its model/);
// A direct-source runtime label names no published identity: its characters are checked, its suffix is not.
assert.equal(resolveConvertedModelId({ explicitModelId: 'some-label', runtimeLabel: true, ...conversion('q4k-ef16-af32') }), 'some-label');
assert.throws(() => resolveConvertedModelId({ explicitModelId: 'some_label', runtimeLabel: true }), /not a valid model name/);
// A label taken from a file's name is brought into line, since nobody chose it.
assert.equal(modelLabelFromFileName('Qwen_Qwen3.6-35B-A3B-Q4_0'), 'qwen-qwen3-6-35b-a3b-q4-0');
assert.ok(isCanonicalModelId(modelLabelFromFileName('__Some  File__(1)')));
assert.equal(modelLabelFromFileName('___'), null);
// A name written by hand before the rule still converts under its old identity.
assert.equal(resolveConvertedModelId({ explicitModelId: 'qwen-3-5-2b-q4k-ehaf16', ...conversion('q4k-ef16-af32') }), 'qwen-3-5-2b-q4k-ehaf16');
assert.throws(() => assertModelIdDescribesFormat('new-2b-q4k-ehaf16', 'q4k-ef16-af32'), /does not describe its model/);

// Every name this repository ships or can produce obeys the rule, apart from the listed earlier identities.
const names = new Map();
const catalog = JSON.parse(fs.readFileSync(path.join(root, 'models', 'catalog.json'), 'utf8'));
for (const entry of catalog.models ?? catalog) names.set(entry.modelId ?? entry.id, 'models/catalog.json');
const walk = dir => fs.readdirSync(dir, { withFileTypes: true }).flatMap(item => item.isDirectory()
  ? walk(path.join(dir, item.name)) : item.name.endsWith('.json') ? [path.join(dir, item.name)] : []);
// A conversion config's name is checked as the converter would emit it.
for (const file of walk(path.join(root, 'src', 'config', 'conversion'))) {
  const config = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (typeof config?.output?.modelBaseId !== 'string' || !config.output.modelBaseId) continue;
  names.set(resolveConvertedModelId({ converterConfig: config }), path.relative(root, file));
}
assert.ok(names.size > 20, 'expected the catalog and the conversion configs to name models');
const offenders = [...names].filter(([id]) => !isCanonicalModelId(id) && !LEGACY_NONCANONICAL_MODEL_IDS.includes(id));
assert.deepEqual(offenders, [], 'model names that break docs/style/model-naming.md');
for (const legacy of LEGACY_NONCANONICAL_MODEL_IDS) assert.ok(names.has(legacy), `${legacy} is listed as an earlier identity but nothing names it; remove it from the list`);

// The suffix of every conversion config's name is the one its own quantization settings generate, apart from
// the listed names written by hand before the rule. This reads the config alone; the converter and the refresh
// tool repeat the check exactly, against the tensors.
const disagreeing = [];
for (const file of walk(path.join(root, 'src', 'config', 'conversion'))) {
  const config = JSON.parse(fs.readFileSync(file, 'utf8'));
  const quantization = config?.quantization ?? {};
  if (typeof config?.output?.modelBaseId !== 'string' || !config.output.modelBaseId || !quantization.weights) continue;
  const modelId = resolveConvertedModelId({ converterConfig: config });
  // Whether a vision, audio or projector part is named depends on the tensors, which a config does not list.
  // So this reads the text part, and accepts further parts after it.
  const info = buildQuantizationInfo(config, quantization.weights, quantization.embeddings ?? null, quantization.lmHead ?? null);
  const text = buildModelNameSuffix(info);
  const index = modelId.indexOf(`-${text}`);
  const describes = index > 0 && /^(-[a-z0-9]+)*$/.test(modelId.slice(index + text.length + 1));
  if (!describes && !isLegacyModelId(modelId)) disagreeing.push(`${modelId}: its suffix must begin with "-${text}"`);
}
assert.deepEqual(disagreeing, [], 'conversion configs whose name does not describe their format');
for (const legacy of LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS) {
  assert.ok(isCanonicalModelId(legacy), `${legacy} belongs in LEGACY_NONCANONICAL_MODEL_IDS`);
  assert.ok(names.has(legacy), `${legacy} is listed as an earlier identity but nothing names it; remove it from the list`);
}

// Check complete format records wherever the catalog retains a source manifest, including multimodal parts.
for (const { modelId } of catalog.models) {
  const filename = path.join(root, 'models', 'local', modelId, 'manifest.json');
  if (!fs.existsSync(filename)) continue;
  const manifest = JSON.parse(fs.readFileSync(filename, 'utf8'));
  assert.equal(manifest.modelId, modelId);
  assert.equal(resolveConvertedModelId({ explicitModelId: modelId, quantizationInfo: manifest.quantizationInfo }), modelId);
}

// quantizationInfo has to be true for the suffix to be: a config that claims an F16 head over a Q4_K_M tensor
// is refused when its manifest is refreshed.
const tensors = { embed: { role: 'embedding', dtype: 'F16' }, head: { role: 'lm_head', dtype: 'Q4_K_M' }, other: { role: 'matmul', dtype: 'Q4_K_M' } };
assert.deepEqual(formatRecordProblems({ weights: 'q4k', embeddings: 'f16', lmHead: 'q4k' }, tensors), []);
assert.deepEqual(formatRecordProblems({ weights: 'q4k', embeddings: 'f16' }, { embed: tensors.embed }), []);
assert.match(formatRecordProblems({ weights: 'q4k', embeddings: 'f16', lmHead: 'f16' }, tensors)[0], /lmHead says f16 but the lm_head tensor is stored as q4k/);
assert.deepEqual(formatRecordProblems({ weights: 'q4_0', embeddings: 'q4_0', lmHead: 'q6k' }, [{ role: 'embedding', dtype: 'Q4_0' }, { role: 'lm_head', dtype: 'Q6_K' }]), []);

// A reissued model keeps every rule its earlier identity had: the rules are keyed by name.
const { resolveCapabilityTransforms } = await import('../../src/rules/capability-transform-resolver.js');
const lanes = id => [true, false].flatMap(hasSubgroups => [true, false].flatMap(hasF16 => ['f16', 'f32', null].map(requested => {
  const resolved = resolveCapabilityTransforms({ hasSubgroups, hasF16 }, { id: 'test', vendor: 'test', architecture: 'test' },
    { modelId: id, requestedActivationDtype: requested, activationDtype: requested ?? 'f32', mathDtype: 'f32', accumDtype: 'f32', kvDtype: 'f16', headDim: 128 });
  return `${hasSubgroups}/${hasF16}/${requested}: ${resolved.kind} ${resolved.names.join(',')}`;
})));
for (const [earlier, reissued] of [['qwen-3-5-2b-q4k-ehaf16', 'qwen-3-5-2b-q4k-ef16-af32'], ['qwen-3-5-0-8b-q4k-ehaf16', 'qwen-3-5-0-8b-q4k-ef16-af32-vf16']]) {
  assert.deepEqual(lanes(reissued), lanes(earlier), `${reissued} resolves different transforms from ${earlier}`);
  assert.ok(lanes(reissued).some(line => line.includes('useQwenF16PrimaryMatmuls')), `${reissued} lost the F16 lane`);
}

console.log('model-id-naming.test: ok');
