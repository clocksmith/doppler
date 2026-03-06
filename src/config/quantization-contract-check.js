import {
  K_SCALE_SIZE,
  Q4K_BLOCK_BYTES,
  Q6K_BLOCK_BYTES,
  Q8_0_BLOCK_BYTES,
  Q8_0_BLOCK_SIZE,
  QK4_K_BLOCK_SIZE,
  QK_K,
  padToQ4KBlock,
  q4kBlockCount,
} from './schema/quantization.schema.js';
import {
  TILE_SIZES,
  QUANTIZATION,
} from '../gpu/kernels/constants.js';
import * as loaderQuantization from '../loader/quantization-constants.js';

const EXPECTED_CONSTANTS = Object.freeze({
  QK_K: 256,
  Q4K_BLOCK_BYTES: 144,
  Q6K_BLOCK_BYTES: 210,
  Q8_0_BLOCK_BYTES: 34,
  Q8_0_BLOCK_SIZE: 32,
  K_SCALE_SIZE: 12,
});

export function buildQuantizationContractArtifact() {
  const errors = [];
  const checks = [];

  const literalConstantsOk =
    QK_K === EXPECTED_CONSTANTS.QK_K
    && Q4K_BLOCK_BYTES === EXPECTED_CONSTANTS.Q4K_BLOCK_BYTES
    && Q6K_BLOCK_BYTES === EXPECTED_CONSTANTS.Q6K_BLOCK_BYTES
    && Q8_0_BLOCK_BYTES === EXPECTED_CONSTANTS.Q8_0_BLOCK_BYTES
    && Q8_0_BLOCK_SIZE === EXPECTED_CONSTANTS.Q8_0_BLOCK_SIZE
    && K_SCALE_SIZE === EXPECTED_CONSTANTS.K_SCALE_SIZE
    && QK4_K_BLOCK_SIZE === Q4K_BLOCK_BYTES;
  if (!literalConstantsOk) {
    errors.push('[QuantizationContract] schema constants drifted from the expected Q4K/Q6K/Q8 values.');
  }
  checks.push({ id: 'quantization.constants.schema', ok: literalConstantsOk });

  const crossModuleOk =
    loaderQuantization.QK_K === QK_K
    && loaderQuantization.Q4K_BLOCK_BYTES === Q4K_BLOCK_BYTES
    && loaderQuantization.Q6K_BLOCK_BYTES === Q6K_BLOCK_BYTES
    && loaderQuantization.Q8_0_BLOCK_BYTES === Q8_0_BLOCK_BYTES
    && loaderQuantization.Q8_0_BLOCK_SIZE === Q8_0_BLOCK_SIZE
    && TILE_SIZES.Q4K_SUPER_BLOCK_SIZE === QK_K
    && QUANTIZATION.Q4K_BLOCK_BYTES === Q4K_BLOCK_BYTES;
  if (!crossModuleOk) {
    errors.push('[QuantizationContract] loader/GPU quantization constants drifted from schema constants.');
  }
  checks.push({ id: 'quantization.constants.crossModule', ok: crossModuleOk });

  let padPropertiesOk = true;
  let q4kCoverageOk = true;
  let previous = -1;
  for (let size = 0; size <= QK_K * 2 + 7; size += 1) {
    const padded = padToQ4KBlock(size);
    if (padded < size || padded % QK_K !== 0 || padToQ4KBlock(padded) !== padded || padded < previous) {
      padPropertiesOk = false;
      break;
    }
    previous = padded;
    if (q4kBlockCount(size) * QK_K < size) {
      q4kCoverageOk = false;
      break;
    }
  }
  if (!padPropertiesOk) {
    errors.push('[QuantizationContract] padToQ4KBlock must be monotone, aligned, and idempotent.');
  }
  checks.push({ id: 'quantization.padToQ4KBlock.properties', ok: padPropertiesOk });
  if (!q4kCoverageOk) {
    errors.push('[QuantizationContract] q4kBlockCount must cover the requested element count.');
  }
  checks.push({ id: 'quantization.q4kBlockCount.coverage', ok: q4kCoverageOk });

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats: {
      sampledSizes: QK_K * 2 + 8,
    },
  };
}
