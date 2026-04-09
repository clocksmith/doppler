import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import {
  DEFAULT_SELF_SPECULATION_CONFIG,
  SPECULATION_MODES,
  SPECULATION_VERIFY_MODES,
  validateSelfSpeculationConfig,
} from '../../src/config/schema/speculation-self.schema.js';

describe('speculation-self schema', () => {
  it('has frozen defaults', () => {
    assert.ok(Object.isFrozen(DEFAULT_SELF_SPECULATION_CONFIG));
    assert.strictEqual(DEFAULT_SELF_SPECULATION_CONFIG.mode, 'none');
    assert.strictEqual(DEFAULT_SELF_SPECULATION_CONFIG.tokens, 1);
    assert.strictEqual(DEFAULT_SELF_SPECULATION_CONFIG.verify, 'greedy');
    assert.strictEqual(DEFAULT_SELF_SPECULATION_CONFIG.threshold, null);
    assert.strictEqual(DEFAULT_SELF_SPECULATION_CONFIG.rollbackOnReject, true);
  });

  it('modes include none, self, draft, medusa', () => {
    assert.deepStrictEqual(SPECULATION_MODES, ['none', 'self', 'draft', 'medusa']);
  });

  it('verify modes include greedy', () => {
    assert.deepStrictEqual(SPECULATION_VERIFY_MODES, ['greedy']);
  });

  it('validates mode=none without error', () => {
    validateSelfSpeculationConfig({ mode: 'none', tokens: 1, verify: 'greedy', threshold: null, rollbackOnReject: true });
  });

  it('validates mode=self tokens=1 without error', () => {
    validateSelfSpeculationConfig({ mode: 'self', tokens: 1, verify: 'greedy', threshold: null, rollbackOnReject: true });
  });

  it('rejects mode=draft with explicit error', () => {
    assert.throws(
      () => validateSelfSpeculationConfig({ mode: 'draft', tokens: 1, verify: 'greedy', threshold: null, rollbackOnReject: true }),
      /mode="draft".*not yet supported/
    );
  });

  it('rejects mode=medusa with explicit error', () => {
    assert.throws(
      () => validateSelfSpeculationConfig({ mode: 'medusa', tokens: 1, verify: 'greedy', threshold: null, rollbackOnReject: true }),
      /mode="medusa".*not yet supported/
    );
  });

  it('rejects unknown mode', () => {
    assert.throws(
      () => validateSelfSpeculationConfig({ mode: 'unknown', tokens: 1, verify: 'greedy', threshold: null, rollbackOnReject: true }),
      /not supported/
    );
  });

  it('validates tokens > 1 for self mode', () => {
    validateSelfSpeculationConfig({ mode: 'self', tokens: 4, verify: 'greedy', threshold: null, rollbackOnReject: true });
  });

  it('rejects tokens=0 for self mode', () => {
    assert.throws(
      () => validateSelfSpeculationConfig({ mode: 'self', tokens: 0, verify: 'greedy', threshold: null, rollbackOnReject: true }),
      /positive integer/
    );
  });

  it('rejects non-object config', () => {
    assert.throws(() => validateSelfSpeculationConfig(null), /non-null object/);
    assert.throws(() => validateSelfSpeculationConfig('self'), /non-null object/);
  });
});
