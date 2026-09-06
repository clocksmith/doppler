import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-source-facts-'));
const run = (...args) => spawnSync(process.execPath, ['tools/forge-source-truth-model-ir-v2.js', ...args], { encoding: 'utf8' });
try {
  const config = Buffer.from('{"model_type":"fixture"}\n');
  const header = Buffer.from(JSON.stringify({ 'embed.weight': { dtype: 'F32', shape: [2, 2], data_offsets: [0, 16] } }));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.length));
  const headerBytes = Buffer.concat([prefix, header]);
  await fs.writeFile(path.join(root, 'config.json'), config);
  await fs.writeFile(path.join(root, 'model.safetensors'), Buffer.concat([headerBytes, Buffer.alloc(16)]));
  const sources = {
    config: { path: path.join(root, 'config.json'), format: 'json', hash: hashBytesSha256(config) },
    headers: { path: path.join(root, 'model.safetensors'), format: 'safetensors-header', hash: hashBytesSha256(headerBytes) },
  };
  const node = (id, fields) => ({ id, factRefs: ['model.type'], ...fields });
  const spec = {
    modelId: 'fixture', sources,
    sourceIdentity: { checkpointId: 'fixture', repository: 'fixture', revision: '1'.repeat(40),
      artifacts: Object.entries(sources).map(([artifactId, source]) => ({ artifactId, path: source.path, role: source.format, hash: source.hash })) },
    defaultAuthorship: { kind: 'tool', actor: 'test-fixture' },
    facts: [
      { id: 'model.type', subject: 'model', predicate: 'type', source: { artifactId: 'config', file: 'config.json', pointer: '/model_type' } },
      { id: 'tensor.embedding', subject: 'embedding', predicate: 'header', source: { kind: 'tensor-header', artifactId: 'headers', file: 'model.safetensors', tensorName: 'embed.weight' } },
    ],
    topology: {
      components: [node('decoder', { type: 'text-decoder', role: 'primary', properties: {} })],
      blockClasses: [node('attention', { kind: 'full-attention', geometry: {}, normalization: {}, positional: {}, feedForward: {}, phaseBehavior: {} })],
      blockSchedules: [node('schedule', { componentId: 'decoder', blocks: [{ index: 0, blockClassId: 'attention' }] })],
      stateSpaces: [node('kv', { kind: 'kv', persistence: 'session', contract: {} })],
      tensorRoleBindings: [node('embedding', { componentId: 'decoder', role: 'embedding', selector: { exact: 'embed.weight' } })],
      entryPoints: [node('generate', { componentId: 'decoder', kind: 'generate', status: 'unlowered', phases: [], reason: 'Synthetic intake, no execution.' })],
      outputHeads: [node('head', { componentId: 'decoder', kind: 'causal-lm' })],
      supportScope: { sourceTopology: 'complete', loweredEntryPoints: [], qualifiedEntryPoints: [], unloweredEntryPoints: ['generate'] },
    },
  };
  const specPath = path.join(root, 'spec.json');
  const output = path.join(root, 'receipt.json');
  await fs.writeFile(specPath, JSON.stringify(spec));
  const result = run('--spec', specPath, '--out', output);
  assert.equal(result.status, 0, result.stderr);
  const receipt = JSON.parse(await fs.readFile(output, 'utf8'));
  assert.deepEqual(receipt.modelIR.provenance.facts[1].value, { shape: [2, 2], dtype: 'F32' });
  assert.deepEqual(receipt.modelIR.supportScope.qualifiedEntryPoints, []);
  const duplicate = run('--spec', specPath, '--out', output);
  assert.notEqual(duplicate.status, 0);
  assert.match(duplicate.stderr, /EEXIST/);
  await fs.appendFile(sources.config.path, ' ');
  const corrupt = run('--spec', specPath, '--out', path.join(root, 'corrupt.json'));
  assert.notEqual(corrupt.status, 0);
  assert.match(corrupt.stderr, /byte hash mismatch/);
  await fs.writeFile(sources.config.path, config);
  await fs.truncate(sources.headers.path, 9);
  const truncated = run('--spec', specPath, '--out', path.join(root, 'truncated.json'));
  assert.notEqual(truncated.status, 0);
  assert.match(truncated.stderr, /Truncated SafeTensors header/);
} finally {
  await fs.rm(root, { recursive: true, force: true });
}
console.log('source-truth-input-files.test: ok (synthetic source intake, no qualification)');
