import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

// Kernels that read weight buffers as raw array<f16> — NOT compatible with Q4K
// block-quantized weights. Q4K packs 4-bit quantized values + scale factors in
// a binary block format; reading those bytes as f16 values produces garbage projections.
const F16_ONLY_WEIGHT_KERNELS = new Set([
  'matmul.f16w.f32a.main',
  'matmul.f16w.f32a.tiled',
]);

// Ops that perform a weight-read matmul and must use a quantization-compatible kernel.
const WEIGHT_READ_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
  'lm_head', 'lm_head_prefill',
]);

function collectManifestPaths(rootDir) {
  const out = [];
  function walk(currentDir) {
    for (const entry of readdirSync(currentDir, { withFileTypes: true })) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
        continue;
      }
      if (entry.isFile() && entry.name === 'manifest.json') {
        out.push(fullPath);
      }
    }
  }
  walk(rootDir);
  out.sort((left, right) => left.localeCompare(right));
  return out;
}

function loadJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

for (const manifestPath of collectManifestPaths(path.join(process.cwd(), 'models'))) {
  const manifest = loadJson(manifestPath);
  const label = path.relative(process.cwd(), manifestPath);
  const inference = manifest?.inference;
  assert.ok(inference && typeof inference === 'object', `${label}: inference is required`);

  if (!Array.isArray(inference.execution?.steps)) {
    continue;
  }

  const session = inference.session;
  assert.ok(session && typeof session === 'object', `${label}: manifests with execution steps require session`);
  assert.ok(
    session.compute?.defaults && typeof session.compute.defaults === 'object',
    `${label}: manifests with execution steps require session.compute.defaults`
  );
  assert.notEqual(
    session.kvcache,
    undefined,
    `${label}: manifests with execution steps require explicit session.kvcache`
  );
  assert.notEqual(
    session.decodeLoop,
    undefined,
    `${label}: manifests with execution steps require explicit session.decodeLoop`
  );
  assert.ok(
    Array.isArray(inference.execution?.steps),
    `${label}: manifests with execution steps require execution.steps`
  );

  // Guard: Q4K manifests using fused Q4K kernels must not assign an F16-only weight kernel
  // to prefill projection ops. matmul_f16w_f32a.wgsl reads weights as array<f16>; Q4K
  // block-quantized bytes are not valid F16 elements — this misread produces garbage projections.
  //
  // Exception: "dequant" kernel paths dequantize Q4K weights to F16 during loading, so
  // F16 matmul kernels are correct in that pipeline — the weights are already F16 in GPU memory.
  const kernelPathId = typeof inference.defaultKernelPath === 'string' ? inference.defaultKernelPath : '';
  const isDequantPipeline = kernelPathId.includes('dequant');
  if (manifest.quantizationInfo?.weights === 'q4k' && !isDequantPipeline) {
    for (const step of inference.execution.steps) {
      if (step.phase !== 'prefill' && step.phase !== 'both') continue;
      if (!WEIGHT_READ_OPS.has(step.op)) continue;
      const kernelRefId = step.kernelRef?.id;
      assert.ok(
        !F16_ONLY_WEIGHT_KERNELS.has(kernelRefId),
        `${label}: Q4K model must not use F16-only kernel '${kernelRefId}' ` +
        `for prefill op '${step.op}' (step '${step.id ?? step.op}'). ` +
        `matmul_f16w_f32a reads array<f16> — Q4K weights are block-quantized, not raw F16. ` +
        `Use a Q4K-dequantizing prefill kernel instead.`
      );
    }
  }
}

console.log('models-manifest-contract.test: ok');
