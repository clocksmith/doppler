import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createSourceTruthPacket, forgeModelIRV2 } from '../../src/converter/source-truth-forge.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

async function forgeCheckedInSpec(name) {
  const specPath = path.join(repoRoot, 'reports/model-ir-v2', `${name}.spec.json`);
  const receiptPath = path.join(repoRoot, 'reports/model-ir-v2', `${name}.model-ir-receipt.json`);
  const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
  const sources = {};
  for (const [artifactId, sourcePath] of Object.entries(spec.sources)) {
    sources[artifactId] = JSON.parse(await fs.readFile(path.join(repoRoot, sourcePath), 'utf8'));
  }
  delete spec.sources;
  const packet = createSourceTruthPacket(spec, sources);
  const receipt = forgeModelIRV2(packet, sources);
  const checkedIn = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
  assert.deepEqual(receipt, checkedIn, `${name} receipt must be deterministically reproducible`);
  return { spec, sources, packet, receipt };
}

// Structural ModelIR invariants that hold for any heterogeneous source, independent of
// the specific model's hyperparameters.
function assertModelIRInvariants(name, receipt) {
  const ir = receipt.modelIR;
  assert.deepEqual(receipt.unresolvedFacts, [], `${name}: signed ModelIR must not carry unresolved facts`);

  const factIds = new Set(ir.provenance.facts.map((fact) => fact.id));
  assert.ok(ir.provenance.facts.every((fact) => (
    fact.disposition === 'accepted'
    && ['direct', 'derived'].includes(fact.confidence)
    && fact.validation.status === 'passed'
    && fact.evidence.length > 0
  )), `${name}: every signed fact must be accepted, sourced and validated`);
  for (const node of [...ir.components, ...ir.blockSchedules, ...ir.stateSpaces, ...ir.entryPoints]) {
    for (const ref of node.factRefs ?? []) {
      assert.ok(factIds.has(ref), `${name}: ${node.id} references unknown fact ${ref}`);
    }
  }

  const componentIds = new Set(ir.components.map((component) => component.id));
  const blockClassIds = new Set(ir.blockClasses.map((blockClass) => blockClass.id));
  for (const schedule of ir.blockSchedules) {
    assert.ok(componentIds.has(schedule.componentId), `${name}: ${schedule.id} targets unknown component`);
    schedule.blocks.forEach((block, index) => {
      assert.equal(block.index, index, `${name}: ${schedule.id} block indexes must be contiguous`);
      assert.ok(blockClassIds.has(block.blockClassId), `${name}: ${schedule.id}[${index}] uses undeclared block class`);
    });
    const component = ir.components.find((entry) => entry.id === schedule.componentId);
    if (Number.isInteger(component.properties?.numLayers)) {
      assert.equal(schedule.blocks.length, component.properties.numLayers,
        `${name}: ${schedule.id} length must match the component's sourced layer count`);
    }
  }

  // Support scope partitions the declared entry points and agrees with each entry point's status.
  const scope = ir.supportScope;
  const entryIds = ir.entryPoints.map((entryPoint) => entryPoint.id);
  const scoped = [...scope.loweredEntryPoints, ...scope.unloweredEntryPoints];
  assert.equal(new Set(scoped).size, scoped.length, `${name}: lowered and unlowered entry points must be disjoint`);
  assert.deepEqual([...scoped].sort(), [...entryIds].sort(), `${name}: support scope must cover every entry point`);
  for (const entryPoint of ir.entryPoints) {
    const expected = scope.loweredEntryPoints.includes(entryPoint.id) ? 'lowered' : 'unlowered';
    assert.equal(entryPoint.status, expected, `${name}: ${entryPoint.id} status must match support scope`);
  }
  for (const qualified of scope.qualifiedEntryPoints) {
    assert.ok(scope.loweredEntryPoints.includes(qualified), `${name}: ${qualified} cannot be qualified before lowering`);
  }
}

const qwen = await forgeCheckedInSpec('qwen3.8-27b');
assertModelIRInvariants('qwen3.8-27b', qwen.receipt);

const glimmer = await forgeCheckedInSpec('glimmer-30b');
assertModelIRInvariants('glimmer-30b', glimmer.receipt);

// Generation defaults are carried from the source generation config, not authored.
{
  const generation = glimmer.sources.generation;
  const textEntryPoint = glimmer.receipt.modelIR.entryPoints.find((entryPoint) => entryPoint.id === 'text.generate');
  assert.deepEqual(textEntryPoint.sourceDefaults, {
    bosTokenId: generation.bos_token_id,
    stopTokenIds: generation.eos_token_id,
    padTokenId: generation.pad_token_id,
    maxLength: generation.max_length,
    sampling: {
      doSample: generation.do_sample,
      temperature: generation.temperature,
      topP: generation.top_p,
      topK: generation.top_k,
    },
  });
}

// Failure paths: inferred facts, missing evidence and tampered source bytes cannot be signed.
{
  const inferred = structuredClone(qwen.packet);
  inferred.facts[0].confidence = 'family-inferred';
  assert.throws(() => forgeModelIRV2(inferred, qwen.sources), /cannot enter signed ModelIR/);

  const unsourced = structuredClone(glimmer.packet);
  unsourced.facts[0].evidence = [];
  assert.throws(() => forgeModelIRV2(unsourced, glimmer.sources), /requires source evidence/);

  const tampered = structuredClone(glimmer.sources);
  tampered.generation.bos_token_id += 1;
  assert.throws(() => forgeModelIRV2(glimmer.packet, tampered), /hash mismatch/);
}

const runtimeFiles = (await fs.readdir(path.join(repoRoot, 'src/client/runtime')))
  .filter((file) => file.endsWith('.js'));
for (const file of runtimeFiles) {
  const source = await fs.readFile(path.join(repoRoot, 'src/client/runtime', file), 'utf8');
  assert.doesNotMatch(source, /qwen|glimmer/i, `${file} must remain model-family agnostic`);
}

console.log('heterogeneous-source-truth.test.js passed');
