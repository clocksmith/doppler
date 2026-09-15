import assert from 'node:assert/strict';
import { buildWgslClosure } from '../../src/tooling/program-bundle/wgsl-closure.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

for (const [file, required] of [['rmsnorm_stats_subgroups.wgsl', ['subgroup_id']], ['rmsnorm_stats.wgsl', []]]) {
  const execution = { kernels: { stats: { kernel: file, entry: 'main',
    digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${file}#main`]}` } }, decode: [['rmsnorm', 'stats']] };
  const closure = await buildWgslClosure(execution, [], { repoRoot: process.cwd() });
  const metadata = closure.modules[0].metadata;
  assert.deepEqual(metadata.requiredWgslFeatures ?? [], required);
  if (!required.length) assert.equal(Object.hasOwn(metadata, 'requiredWgslFeatures'), false,
    'absent language requirements do not change historical metadata shape');
  const { sourceMetadataHash, ...fields } = metadata;
  assert.equal(sourceMetadataHash, computeCanonicalSha256(fields));
}
console.log('wgsl-language-export.test: source requirements are bound by metadata identity');
