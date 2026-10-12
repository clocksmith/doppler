// Builds an isolated source experiment; never edits the canonical runtime tree.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
const root = resolve(process.argv[2] ?? '');
assert(root.includes('/reports/local/source-prefix-correction-20261012/build'));
const replace = async (path, from, to) => {
  const file = join(root, path), source = await readFile(file, 'utf8');
  assert.equal(source.split(from).length, 2, path);
  await writeFile(file, source.replace(from, to));
};
await replace('src/inference/pipelines/text/attention/interpreter.js',
  '      normalizationAxis: config.queryKeyNormAxis,',
  `      normalizationAxis: config.queryKeyNormAxis,
      // Only unchanged coordinates going directly to this declared F16 cache
      // may defer their rounding. Rotated/shared/aliased values stay ordinary.
      kHalfRoundingStart: !skipKVCacheWrites && !retainSharedKvBuffers
        && !valueAliasesKey && !diffusionGemmaDecoder
        && state.kvCache?.kvDtype === 'f16' && qTensor.dtype === 'f32'
        && state.kvCache.hasGPUCache?.() === true
        && config.queryKeyNormAxis === 'head'
        && (config.ropeFrequencyBaseDim ?? config.ropeRotaryDim) === config.ropeRotaryDim
        ? (disableRoPE ? 0 : config.ropeRotaryDim) : null,`);
await replace('src/inference/pipelines/text/attention/projection-plan.js',
  "  normalizationAxis = 'head',", "  normalizationAxis = 'head',\n  kHalfRoundingStart = null,");
await replace('src/inference/pipelines/text/attention/projection-plan.js',
  '        rmsNormWeightOffset: effectiveRmsNormWeightOffset,\n      });',
  '        rmsNormWeightOffset: effectiveRmsNormWeightOffset,\n        kHalfRoundingStart,\n      });');
await replace('src/gpu/kernels/rmsnorm-qk.js',
  '    rmsNormWeightOffset = false,', '    rmsNormWeightOffset = false,\n    kHalfRoundingStart = null,');
await replace('src/gpu/kernels/rmsnorm-qk.js',
  "  assertPositiveInteger(headDim, 'headDim');",
  `  assertPositiveInteger(headDim, 'headDim');
  if (kHalfRoundingStart !== null && (!Number.isInteger(kHalfRoundingStart)
    || kHalfRoundingStart < 0 || kHalfRoundingStart > headDim || kTensor.dtype !== 'f32')) {
    throw new Error('Deferred K rounding requires explicit F32 head coordinates for a subsequent F16 store.');
  }`);
await replace('src/gpu/kernels/rmsnorm-qk.js',
  "        K_WEIGHT_IS_F16: kWeightDtype === 'f16',",
  `        K_WEIGHT_IS_F16: kWeightDtype === 'f16',
        ...(kHalfRoundingStart === null ? {} : {
          DEFER_K_HALF_ROUNDING: true, HALF_ROUNDING_START: kHalfRoundingStart,
        }),`);
await replace('src/gpu/kernels/rmsnorm-qk.d.ts', '  rmsNormWeightOffset?: boolean;',
  '  rmsNormWeightOffset?: boolean;\n  kHalfRoundingStart?: number | null;');
const shader = await readFile(new URL('./candidate-qknorm-deferred-half.wgsl', import.meta.url), 'utf8');
await writeFile(join(root, 'src/gpu/kernels/rmsnorm_qk.wgsl'),
  shader.replace('override DEFER_K_HALF_ROUNDING:', 'override HALF_ROUNDING_START: u32 = 0u;\noverride DEFER_K_HALF_ROUNDING:')
    .replace('DEFER_K_HALF_ROUNDING && row >= u.q_rows', 'DEFER_K_HALF_ROUNDING && row >= u.q_rows && idx >= HALF_ROUNDING_START'));
