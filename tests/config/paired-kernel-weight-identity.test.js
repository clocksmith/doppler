import assert from 'node:assert/strict';
import { kernelPathRequiresWeightDtype, getKernelPathMatmulVariant } from '../../src/config/kernel-path-loader.js';
for (const weight of ['f32', 'f16', 'q4k']) {
  const path = { decode: { steps: [{ op: 'q_proj', kernel: 'matmul_paired.wgsl', entry: 'main', constants: {
    WEIGHT_F16: weight === 'f16', WEIGHT_Q4K: weight === 'q4k', SCALAR_OUTPUT: false,
  } }] } };
  assert.equal(getKernelPathMatmulVariant('q_proj', 'decode', 0, path), `paired_${weight}`);
  for (const candidate of ['f32', 'f16', 'q4k']) assert.equal(kernelPathRequiresWeightDtype(path, candidate, 'matmul'), candidate === weight,
    'Shared WGSL file must not authorize every weight representation; exact declared overrides select one.');
}
console.log('paired kernel weight identity: exact overrides select one materialization');
