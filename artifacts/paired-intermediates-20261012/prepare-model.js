// Standard archive plus unchanged model pieces and an explicitly frozen graph.
import { readFile, writeFile, readdir, mkdir, symlink } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { gunzipSync } from 'node:zlib';
import { bindPairedModel } from '../../tests/helpers/paired-model-binding.js';
const [archive, model, output] = process.argv.slice(2).map(value => resolve(value));
const hash = value => createHash('sha256').update(value).digest('hex');
await mkdir(output, { recursive: true });
execFileSync('tar', ['-xzf', archive, '-C', output]);
const destination = join(output, 'model'); await mkdir(destination);
for (const name of await readdir(model)) if (name !== 'manifest.json') await symlink(join(model, name), join(destination, name));
const source = await readFile(join(model, 'manifest.json'));
const manifest = JSON.parse(source);
const graphBytes = await readFile(new URL('./execution.json', import.meta.url));
manifest.inference.execution = JSON.parse(graphBytes);
for (const kernel of Object.values(manifest.inference.execution.kernels)) {
  const shader = await readFile(join(output, 'package/src/gpu/kernels', kernel.kernel), 'utf8');
  if (kernel.digest !== 'sha256:' + hash(shader.replaceAll('\r\n', '\n') + '\n@@entry:' + kernel.entry)) {
    throw new Error(`Frozen graph does not bind packaged shader ${kernel.kernel}#${kernel.entry}`);
  }
}
const baselineBytes = gunzipSync(await readFile(new URL('../../../reploid/tests/fixtures/distributed-reference-manifest.json.gz', import.meta.url)));
const scope = bindPairedModel(manifest, JSON.parse(baselineBytes), JSON.parse(graphBytes));
const bytes = JSON.stringify(manifest, null, 2) + '\n';
await writeFile(join(destination, 'manifest.json'), bytes);
const binding = { archiveSha256: hash(await readFile(archive)), sourceManifestSha256: hash(baselineBytes),
  immediateParentManifestSha256: hash(source), manifestSha256: hash(bytes), graphSha256: hash(graphBytes),
  representation: 'Explicit session intermediatePrecision=two-f32 and linearRecurrentStatePrecision=two-f32',
  ...scope, qualified: false, sourceConstruction: 'standard npm pack from canonical source; no archive shader substitution' };
await writeFile(join(output, 'binding.json'), JSON.stringify(binding, null, 2) + '\n');
console.log(JSON.stringify(binding));
