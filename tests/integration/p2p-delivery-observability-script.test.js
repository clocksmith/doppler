import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const {
  runP2PDeliveryObservabilityCli,
} = await import('../../tools/p2p-delivery-observability.js');

const tempDir = mkdtempSync(join(tmpdir(), 'doppler-p2p-obsv-'));
const inputPath = join(tempDir, 'metrics.json');

try {
  const payload = [
    {
      deliveryMetrics: {
        totalDurationMs: 100,
        successSource: 'p2p',
        attemptCount: 1,
        sourceAttempts: { cache: 0, p2p: 1, http: 0 },
        failureCodes: {},
      },
    },
    {
      deliveryMetrics: {
        totalDurationMs: 400,
        successSource: 'http',
        attemptCount: 2,
        sourceAttempts: { cache: 0, p2p: 1, http: 1 },
        failureCodes: {
          DOPPLER_DISTRIBUTION_P2P_TRANSPORT_UNAVAILABLE: 1,
        },
      },
    },
  ];
  writeFileSync(inputPath, JSON.stringify(payload), 'utf8');

  const parsed = runP2PDeliveryObservabilityCli([
    '--input',
    inputPath,
    '--json',
  ]);
  assert.ok(parsed?.summary);
  assert.equal(parsed?.summary?.totals?.records, 2);
  assert.ok(Array.isArray(parsed?.alerts));

  console.log('p2p-delivery-observability-script.test: ok');
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}
