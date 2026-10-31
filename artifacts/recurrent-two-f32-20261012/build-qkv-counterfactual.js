// Test-only causal intervention; neither archive is admissible for qualification.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, join } from 'node:path';

const [archive, fixturePath, modelPath, output, mode] = process.argv.slice(2);
assert(['captured-control', 'independent-local'].includes(mode));
const hash = value => createHash('sha256').update(value).digest('hex');
const parentArchiveSha256 = hash(await readFile(archive));
assert.equal(parentArchiveSha256, 'd0d53e8f6d72ef7de63433d5935b2e89a717d3b4d6e8d890bd19add499e5e85e');
const fixtureBytes = await readFile(fixturePath);
const fixture = JSON.parse(fixtureBytes);
assert.equal(fixture.captureSha256, 'aeb905a22459570fb51322e4b9e91dc4b603f236bcfd6507e005dd1a9c271b3c');
assert.equal(fixture.operation, 'layer.0.attn.qkv_proj');
assert.equal(fixture.M, 184);
const reference = Buffer.from(fixture.oracleFloat64, 'base64');
const values = mode === 'captured-control' ? Buffer.from(fixture.baseline026, 'base64')
  : Buffer.alloc(reference.byteLength / 2);
if (mode === 'independent-local') {
  for (let i = 0; i < reference.byteLength / 8; i++) values.writeFloatLE(reference.readDoubleLE(i * 8), i * 4);
}
assert.equal(values.byteLength, fixture.M * fixture.N * 4);
await mkdir(output);
await promisify(execFile)('tar', ['-xzf', resolve(archive), '-C', resolve(output)]);
const root = resolve(output, 'package');
const path = join(root, 'src/inference/pipelines/text/probes.js');
let source = await readFile(path, 'utf8');
const anchor = "  const { layerIdx, numTokens, hiddenSize, probes, recorder, dtype = 'f32' } = options;";
assert.equal(source.split(anchor).length, 2);
source = source.replace(anchor, anchor + `
  // Retained diagnostic intervention, never a production compute path.
  if (stage === 'linear_qkv_proj' && layerIdx === 0 && numTokens === 184) {
    if (dtype !== 'f32' || hiddenSize !== ${fixture.N}) throw new Error('Counterfactual geometry mismatch');
    const bytes = Uint8Array.from(atob('${values.toString('base64')}'), c => c.charCodeAt(0));
    const device = getDevice();
    if (recorder) {
      const staging = device.createBuffer({ label: 'qkv_counterfactual', size: bytes.byteLength,
        usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      new Uint8Array(staging.getMappedRange()).set(bytes); staging.unmap();
      recorder.getEncoder().copyBufferToBuffer(staging, 0, buffer, 0, bytes.byteLength);
      recorder.trackTemporaryBuffer(staging);
    } else { device.queue.writeBuffer(buffer, 0, bytes); }
    console.log('producer-progress:' + JSON.stringify({ phase: 'counterfactual', mode: '${mode}',
      layer: 0, numTokens, outputSha256: '${hash(values)}', qualification: false }));
  }`);
await writeFile(path, source);
const pack = await promisify(execFile)('npm', ['pack', '--ignore-scripts', '--pack-destination', resolve(output)], { cwd: root });
await writeFile(join(output, 'pack.log'), pack.stdout + pack.stderr);
const file = pack.stdout.trim().split('\n').at(-1);
const binding = { archiveSha256: hash(await readFile(join(output, file))), parentArchiveSha256,
  manifestSha256: hash(await readFile(join(modelPath, 'manifest.json'))),
  sourceManifestSha256: '27b7363a12c7448fe6f71d333ba9ee944dc7a789ce357c6b98a2a02c1b44a281',
  fixtureSha256: hash(fixtureBytes), captureSha256: fixture.captureSha256,
  injectedOutputSha256: hash(values), mode, unchangedWeights: true, tolerance: 0.001,
  scope: 'Only layer0 prefill QKV overwritten; actual downstream GPU, never qualification' };
await writeFile(join(output, 'binding.json'), JSON.stringify(binding, null, 2) + '\n');
console.log(JSON.stringify(binding));
