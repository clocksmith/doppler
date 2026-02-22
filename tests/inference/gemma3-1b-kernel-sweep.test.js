import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const KERNEL_PATHS_DIR = resolve(__dirname, '../../src/config/presets/kernel-paths');
const REGISTRY_FILE = resolve(KERNEL_PATHS_DIR, 'registry.json');

function loadKernelPath(id) {
  return JSON.parse(readFileSync(resolve(KERNEL_PATHS_DIR, `${id}.json`), 'utf8'));
}

// The four paths that constitute the Gemma 3 1b kernel sweep (f16a, f32a, Q4K variants).
const SWEEP = [
  'gemma3-f16-fused-f32a-online',
  'gemma3-f16-fused-f16a-online',
  'gemma3-q4k-dequant-f32a-online',
  'gemma3-q4k-dequant-f16a-online',
];

// Kernels valid for M=1 decode projection ops (subgroup-optimized GEMV).
const SUBGROUP_GEMV_KERNELS = new Set([
  'matmul_gemv_subgroup.wgsl',
  'matmul_gemv_subgroup_f16a.wgsl',
]);

// Kernels valid for online decode attention.
const ONLINE_ATTN_KERNELS = new Set([
  'attention_decode_online_f16kv.wgsl',
  'attention_decode_online_f16.wgsl',
]);

// Individual projection ops that must use subgroup GEMV in decode.
// ffn_gate_up is excluded: fused paths replace gate+up with a single fused op.
const PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
]);

// Tiled matmul kernels that must NOT appear in decode projection ops.
const TILED_MATMUL_KERNELS = new Set([
  'matmul_f16w_f32a.wgsl',
  'matmul_f16_tiled.wgsl',
  'matmul_f16.wgsl',
]);

// 1. All sweep paths are registered in registry.json.
const registry = JSON.parse(readFileSync(REGISTRY_FILE, 'utf8'));
const registeredIds = new Set(registry.entries.map(e => e.id));
for (const pathId of SWEEP) {
  assert.ok(registeredIds.has(pathId), `registry.json missing entry for "${pathId}"`);
}

for (const pathId of SWEEP) {
  const spec = loadKernelPath(pathId);

  // 2. Every decode projection op uses a subgroup GEMV kernel.
  for (const step of spec.decode.steps) {
    if (!PROJECTION_OPS.has(step.op)) continue;
    assert.ok(
      SUBGROUP_GEMV_KERNELS.has(step.kernel),
      `${pathId}: decode "${step.op}" uses "${step.kernel}" — must be subgroup GEMV for M=1 decode`
    );
    assert.equal(
      step.entry, 'main_vec4',
      `${pathId}: decode "${step.op}" entry must be "main_vec4" (got "${step.entry}")`
    );
  }

  // 3. No tiled matmul appears in decode projection ops.
  for (const step of spec.decode.steps) {
    if (!PROJECTION_OPS.has(step.op)) continue;
    assert.ok(
      !TILED_MATMUL_KERNELS.has(step.kernel),
      `${pathId}: decode "${step.op}" uses tiled matmul "${step.kernel}" — regressed from subgroup GEMV`
    );
  }

  // 4. Decode attention uses an online kernel.
  const attnStep = spec.decode.steps.find(s => s.op === 'attention');
  assert.ok(attnStep, `${pathId}: no "attention" step in decode`);
  assert.ok(
    ONLINE_ATTN_KERNELS.has(attnStep.kernel),
    `${pathId}: decode attention uses "${attnStep.kernel}" — expected online kernel`
  );

  // 5. lm_head decode uses main_multicol with tuned MULTICOL constants.
  const lmHead = spec.postLayer.find(s => s.op === 'lm_head');
  assert.ok(lmHead, `${pathId}: no "lm_head" in postLayer`);
  assert.equal(lmHead.entry, 'main_multicol',
    `${pathId}: lm_head entry must be "main_multicol" (got "${lmHead.entry}")`
  );
  const cols = lmHead.constants?.MULTICOL_COLS_PER_WG;
  const threads = lmHead.constants?.MULTICOL_THREADS_PER_COL;
  assert.ok(
    Number.isInteger(cols) && cols >= 64,
    `${pathId}: lm_head MULTICOL_COLS_PER_WG must be >= 64 (got ${cols})`
  );
  assert.equal(
    cols * threads, 256,
    `${pathId}: MULTICOL_COLS_PER_WG (${cols}) * MULTICOL_THREADS_PER_COL (${threads}) must equal workgroup size 256`
  );

  // 6. Prefill projection ops do not use GEMV kernels (tiled matmul is correct for batched rows).
  for (const step of spec.prefill.steps) {
    if (!PROJECTION_OPS.has(step.op)) continue;
    assert.ok(
      !SUBGROUP_GEMV_KERNELS.has(step.kernel),
      `${pathId}: prefill "${step.op}" uses GEMV kernel "${step.kernel}" — GEMV is not valid for batched prefill`
    );
  }
}

console.log('gemma3-1b-kernel-sweep.test: ok');
