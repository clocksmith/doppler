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

  const blockRelationshipsOk =
    Q4K_BLOCK_BYTES === 2 + 2 + K_SCALE_SIZE + (QK_K / 2)
    && Q6K_BLOCK_BYTES === (QK_K / 2) + (QK_K / 4) + (QK_K / 16) + 2
    && Q8_0_BLOCK_BYTES === 2 + Q8_0_BLOCK_SIZE;
  if (!blockRelationshipsOk) {
    errors.push('[QuantizationContract] block byte sizes do not match their declared format components.');
  }
  checks.push({ id: 'quantization.constants.blockRelationships', ok: blockRelationshipsOk });

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
