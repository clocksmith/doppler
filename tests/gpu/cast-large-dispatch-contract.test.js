import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [castSource, shaderSource] = await Promise.all([
  readFile(new URL('../../src/gpu/kernels/cast.js', import.meta.url), 'utf8'),
  readFile(new URL('../../src/gpu/kernels/cast_f16_to_f32.wgsl', import.meta.url), 'utf8'),
]);

assert.match(
  castSource,
  /\[maxWorkgroupsPerDim, Math\.ceil\(workgroups \/ maxWorkgroupsPerDim\), 1\]/,
  'large casts must use a two-dimensional dispatch'
);
assert.match(
  shaderSource,
  /@builtin\(num_workgroups\)\s+num_wg:\s*vec3<u32>/,
  'the f16-to-f32 shader must receive the two-dimensional dispatch geometry'
);
assert.match(
  shaderSource,
  /gid\.x\s*\+\s*gid\.y\s*\*\s*num_wg\.x\s*\*\s*WORKGROUP_SIZE/,
  'the f16-to-f32 shader must flatten both dispatch dimensions'
);

console.log('cast-large-dispatch-contract.test: ok');
