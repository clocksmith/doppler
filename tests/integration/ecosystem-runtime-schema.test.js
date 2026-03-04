import assert from 'node:assert/strict';

const { createDopplerConfig } = await import('../../src/config/schema/doppler.schema.js');
const { validateRuntimeConfig } = await import('../../src/config/param-validator.js');
const {
  DEFAULT_ECOSYSTEM_CONFIG,
  createEcosystemConfig,
  validateEcosystemConfig,
} = await import('../../src/config/schema/ecosystem.schema.js');

{
  validateEcosystemConfig(DEFAULT_ECOSYSTEM_CONFIG);
}

{
  const ecosystem = createEcosystemConfig({
    discovery: {
      ranking: {
        signals: {
          qualityEvidenceWeight: 0.4,
          trustWeight: 0.2,
          compatibilityWeight: 0.2,
          adoptionWeight: 0.2,
        },
      },
    },
    reliability: {
      failover: {
        tiers: ['peer', 'relay', 'origin'],
      },
    },
  });

  validateEcosystemConfig(ecosystem);
  assert.equal(ecosystem.discovery.ranking.signals.qualityEvidenceWeight, 0.4);
  assert.deepEqual(ecosystem.reliability.failover.tiers, ['peer', 'relay', 'origin']);
}

{
  const config = createDopplerConfig({
    runtime: {
      shared: {
        ecosystem: {
          hostedAccess: {
            enabled: true,
            billing: {
              meteringEnabled: true,
            },
          },
          discovery: {
            ranking: {
              mode: 'blended',
            },
          },
        },
      },
    },
  });

  validateRuntimeConfig(config.runtime);
  assert.equal(config.runtime.shared.ecosystem.hostedAccess.enabled, true);
  assert.equal(config.runtime.shared.ecosystem.hostedAccess.billing.meteringEnabled, true);
  assert.equal(config.runtime.shared.ecosystem.discovery.ranking.mode, 'blended');
}

{
  const badConfig = createDopplerConfig({
    runtime: {
      shared: {
        ecosystem: {
          reliability: {
            failover: {
              tiers: ['peer', 'relay'],
            },
          },
        },
      },
    },
  });

  assert.throws(
    () => validateRuntimeConfig(badConfig.runtime),
    /must include "origin"/
  );
}

console.log('ecosystem-runtime-schema.test: ok');
