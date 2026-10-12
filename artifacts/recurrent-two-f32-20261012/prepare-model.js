// Bind unchanged model pieces to the canonical package's declared shaders.
import { readFile, writeFile, readdir, mkdir, symlink } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { compareReferenceModel } from '../../../reploid/tests/fixtures/distributed-reference-model.js';
const [archive, model, output] = process.argv.slice(2).map(value => resolve(value));
const hash = value => createHash('sha256').update(value).digest('hex');
await mkdir(output, { recursive: true });
execFileSync('tar', ['-xzf', archive, '-C', output]);
const destination = join(output, 'model'); await mkdir(destination);
for (const name of await readdir(model)) if (name !== 'manifest.json') await symlink(join(model, name), join(destination, name));
const source = await readFile(join(model,'manifest.json'));
const manifest = JSON.parse(source);
for (const kernel of Object.values(manifest.inference.execution.kernels)) {
  const shader = await readFile(join(output,'package/src/gpu/kernels',kernel.kernel),'utf8');
  kernel.digest = 'sha256:'+hash(shader.replaceAll('\r\n','\n')+'\n@@entry:'+kernel.entry);
}
const bytes = JSON.stringify(manifest,null,2)+'\n';
await writeFile(join(destination,'manifest.json'),bytes);
const referenceContract = JSON.parse(await readFile(new URL('../whole-model-f64-complete-20261011/request-2-contract.json', import.meta.url)));
const approvedSource = await readFile(resolve(referenceContract.inputs.manifest.path));
if (hash(approvedSource) !== referenceContract.inputs.manifest.sha256) throw new Error('Approved source manifest identity changed.');
compareReferenceModel(manifest, JSON.parse(approvedSource));
const binding = { archiveSha256:hash(await readFile(archive)), sourceManifestSha256:hash(approvedSource),
  immediateParentManifestSha256:hash(source), manifestSha256:hash(bytes),
  representation: 'inference.session.linearRecurrentStatePrecision=two-f32 (explicit runtime override)',
  precisionScope:'persistent recurrent state and its compensation only; activation/output F32 and KV F16 unchanged',
  modelPieceBytesUnchanged:true, reference:'approved F64', tolerance:0.001 };
await writeFile(join(output,'binding.json'),JSON.stringify(binding,null,2)+'\n');
console.log(JSON.stringify(binding));
