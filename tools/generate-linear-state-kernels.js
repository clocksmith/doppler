#!/usr/bin/env node
import { readFile, writeFile } from 'node:fs/promises';

const kernelRoot = new URL('../src/gpu/kernels/', import.meta.url);
const helpers = await readFile(new URL('codegen/includes/compensated-f32.wgsl.inc', kernelRoot), 'utf8');
const begin = '// BEGIN GENERATED COMPENSATED STATE\n';
const end = '// END GENERATED COMPENSATED STATE\n';
const kernels = [
  ...['gated_delta_recurrent', 'gated_delta_fused_decode']
    .flatMap(base => ['', '_f16'].map(suffix => `${base}${suffix}.wgsl`)),
  'matmul_compensated_f32.wgsl',
  'matmul_gemv_compensated_f32.wgsl',
];
for (const name of kernels) {
  const path = new URL(name, kernelRoot);
  const source = await readFile(path, 'utf8');
  const start = source.indexOf(begin);
  const stop = source.indexOf(end);
  if (start < 0 || stop < start) throw new Error(`Missing generated state markers: ${path}`);
  const next = source.slice(0, start) + begin + helpers + end + source.slice(stop + end.length);
  if (process.argv.includes('--check')) {
    if (next !== source) throw new Error(`Stale compensated state helpers: ${path}`);
  } else {
    await writeFile(path, next);
  }
}
