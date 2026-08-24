import { log } from '../../../debug/index.js';
import { runVisionPatchEmbed } from '../../../gpu/kernels/vision-patch-embed.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision patch embedding requires ${label} to be a positive integer.`);
  }
}

export async function patchEmbed(params) {
  const {
    imageData,
    height,
    width,
    channels,
    gridHeight,
    gridWidth,
    visionConfig,
    weights,
  } = params;
  const { patchSize, hiddenSize, temporalPatchSize } = visionConfig;
  for (const [label, value] of Object.entries({
    height,
    width,
    channels,
    gridHeight,
    gridWidth,
    patchSize,
    hiddenSize,
    temporalPatchSize,
  })) {
    requirePositiveInteger(value, label);
  }
  if (height !== gridHeight * patchSize || width !== gridWidth * patchSize) {
    throw new Error(
      `Vision patch geometry mismatch: image=${width}x${height}, ` +
      `grid=${gridWidth}x${gridHeight}, patchSize=${patchSize}.`
    );
  }

  const patchProjWeight = weights.patchProjWeight ?? weights['visual.patch_embed.proj.weight'];
  const patchProjBias = weights.patchProjBias ?? weights['visual.patch_embed.proj.bias'] ?? null;
  if (!patchProjWeight) {
    throw new Error(
      'Vision patch embedding weight is missing. ' +
      'Expected weights.patchProjWeight or weights["visual.patch_embed.proj.weight"].'
    );
  }

  const embedded = await runVisionPatchEmbed(
    imageData,
    patchProjWeight,
    patchProjBias,
    {
      gridHeight,
      gridWidth,
      channels,
      patchSize,
      temporalPatchSize,
      hiddenSize,
    }
  );
  const numPatches = gridHeight * gridWidth;
  log.debug(
    'Vision',
    `patchEmbed: ${height}x${width} -> ${gridHeight}x${gridWidth} = ${numPatches} patches (${hiddenSize}d)`
  );
  return { patchBuffer: embedded.buffer, numPatches };
}
