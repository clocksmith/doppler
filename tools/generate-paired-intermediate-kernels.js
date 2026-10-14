#!/usr/bin/env node
import { readFile, writeFile } from 'node:fs/promises';
const root = new URL('../src/gpu/kernels/', import.meta.url);
const source = await readFile(new URL('codegen/includes/paired-half-store.wgsl.inc', root), 'utf8');
const file = new URL('cast_two_f32_to_f16.wgsl', root);
const current = await readFile(file, 'utf8');
const begin = '// BEGIN GENERATED PAIRED HALF\n';
const end = '// END GENERATED PAIRED HALF\n';
const start = current.indexOf(begin), stop = current.indexOf(end);
if (start < 0 || stop < start) throw new Error('Missing paired-half generation markers.');
const expected = current.slice(0, start) + begin + source + end + current.slice(stop + end.length);
if (process.argv.includes('--check')) {
  if (expected !== current) throw new Error('Stale generated paired-half shader.');
} else await writeFile(file, expected);

const arithmetic = (await Promise.all(['compensated-f32', 'paired-elementary'].map(name =>
  readFile(new URL(`codegen/includes/${name}.wgsl.inc`, root), 'utf8')))).join('\n');
for (const name of ['attention_paired.wgsl', 'attention_paired_tiled.wgsl', 'rope_precompute_paired.wgsl', 'rope_qk_paired.wgsl', 'gated_delta_conv_paired.wgsl', 'gated_delta_recurrent_paired.wgsl', 'gated_delta_qk_paired.wgsl', 'gated_delta_state_paired.wgsl', 'gated_delta_norm_paired.wgsl', 'gated_delta_gate_paired.wgsl', 'qwen_linear_attention_prepare_paired.wgsl', 'qwen_linear_attention_scale_paired.wgsl', 'qwen_linear_attention_log_decay_paired.wgsl', 'qwen_linear_attention_decay_paired.wgsl', 'rmsnorm_paired.wgsl', 'rmsnorm_paired_sliced.wgsl', 'matmul_paired.wgsl', 'paired_add.wgsl', 'paired_scale.wgsl', 'paired_scale_inplace.wgsl', 'paired_multiply.wgsl', 'paired_silu.wgsl', 'paired_silu_unary.wgsl', 'paired_sigmoid.wgsl']) {
  const path = new URL(name, root), body = await readFile(path, 'utf8');
  const begin = '// BEGIN GENERATED PAIRED ARITHMETIC\n', end = '// END GENERATED PAIRED ARITHMETIC\n';
  const start = body.indexOf(begin), stop = body.indexOf(end);
  if (start < 0 || stop < start) throw new Error(`Missing paired arithmetic markers: ${name}`);
  const expected = body.slice(0, start) + begin + arithmetic + end + body.slice(stop + end.length);
  if (process.argv.includes('--check')) {
    if (expected !== body) throw new Error(`Stale generated paired arithmetic: ${name}`);
  } else await writeFile(path, expected);
}
