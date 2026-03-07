import assert from 'node:assert/strict';

import {
  TOOLING_ENVELOPE_SCHEMA_VERSION,
  ToolingCommandError,
  createToolingErrorEnvelope,
  createToolingSuccessEnvelope,
  isToolingErrorEnvelope,
  isToolingSuccessEnvelope,
  normalizeToToolingCommandError,
} from '../../src/tooling/command-envelope.js';

{
  const request = {
    command: 'verify',
    suite: 'kernels',
  };
  const envelope = createToolingSuccessEnvelope({
    surface: 'node',
    request,
    result: { passed: 12, failed: 0 },
  });
  assert.equal(envelope.ok, true);
  assert.equal(envelope.schemaVersion, TOOLING_ENVELOPE_SCHEMA_VERSION);
  assert.equal(envelope.surface, 'node');
  assert.deepEqual(envelope.request, request);
  assert.deepEqual(envelope.result, { passed: 12, failed: 0 });
  assert.equal(isToolingSuccessEnvelope(envelope), true);
}

{
  const normalized = normalizeToToolingCommandError(
    new Error('node webgpu missing'),
    {
      surface: 'node',
      request: {
        command: 'verify',
        suite: 'inference',
      },
    }
  );
  assert.equal(normalized.name, 'ToolingCommandError');
  assert.equal(normalized.code, 'tooling_error');
  assert.equal(normalized.details?.surface, 'node');
  assert.equal(normalized.details?.command, 'verify');
  assert.equal(normalized.details?.suite, 'inference');
}

{
  const original = new ToolingCommandError('relay failed', {
    code: 'relay_failed',
    details: {
      fromSurface: 'node',
      toSurface: 'browser',
    },
  });
  const enriched = normalizeToToolingCommandError(original, {
    surface: 'browser',
    request: {
      command: 'bench',
      suite: 'bench',
      workloadType: 'training',
    },
  });
  assert.equal(enriched, original);
  assert.equal(enriched.details?.surface, 'browser');
  assert.equal(enriched.details?.workloadType, 'training');

  const errorEnvelope = createToolingErrorEnvelope(enriched, {
    surface: 'browser',
    request: {
      command: 'bench',
      suite: 'bench',
      workloadType: 'training',
    },
  });
  assert.equal(errorEnvelope.ok, false);
  assert.equal(errorEnvelope.schemaVersion, TOOLING_ENVELOPE_SCHEMA_VERSION);
  assert.equal(errorEnvelope.surface, 'browser');
  assert.equal(errorEnvelope.error.code, 'relay_failed');
  assert.equal(isToolingErrorEnvelope(errorEnvelope), true);
}

{
  const errorEnvelope = createToolingErrorEnvelope(
    new ToolingCommandError('node failed', {
      details: {
        surface: 'node',
        command: 'verify',
        suite: 'inference',
      },
    }),
    {
      surface: null,
      request: {
        command: 'verify',
        suite: 'inference',
      },
    }
  );
  assert.equal(errorEnvelope.surface, 'node');
  assert.equal(errorEnvelope.error.details?.surface, 'node');
}

console.log('command-envelope.test: ok');
