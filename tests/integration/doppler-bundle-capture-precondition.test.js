// Unit-level test for the bundle CLI's capture-precondition check. Forces the
// negative paths for both surfaces so the typed blocker codes and actionable
// messages are covered even when the local host has working bindings.

import assert from 'node:assert/strict';
import { checkCapturePrecondition } from '../../src/cli/doppler-cli.js';

// surface=node with a DOPPLER_NODE_WEBGPU_MODULE pointing at a nonexistent
// specifier must fall into the typed `capture_precondition_node_webgpu_unavailable`
// blocker with an actionable message that names both the npm install hint and
// the --surface browser / --skip-capture alternatives.
{
  const previous = process.env.DOPPLER_NODE_WEBGPU_MODULE;
  process.env.DOPPLER_NODE_WEBGPU_MODULE = '/this/module/does/not/exist.js';
  try {
    const result = await checkCapturePrecondition('node');
    assert.equal(result.ok, false, 'expected precondition to fail for broken node binding');
    assert.equal(result.surface, 'node');
    assert.equal(
      result.code,
      'capture_precondition_node_webgpu_unavailable',
      'blocker code must be typed for agent tooling consumption'
    );
    assert.equal(typeof result.message, 'string');
    assert.match(
      result.message,
      /npm install webgpu|DOPPLER_NODE_WEBGPU_MODULE/,
      'message must surface a concrete install hint'
    );
    assert.match(
      result.message,
      /--surface browser|npx playwright install/,
      'message must surface the browser alternative'
    );
    assert.match(
      result.message,
      /--skip-capture/,
      'message must surface the --skip-capture alternative'
    );
    assert.equal(typeof result.detail, 'string');
  } finally {
    if (previous === undefined) {
      delete process.env.DOPPLER_NODE_WEBGPU_MODULE;
    } else {
      process.env.DOPPLER_NODE_WEBGPU_MODULE = previous;
    }
  }
}

// surface=unsupported must fall into the typed unsupported-surface blocker.
{
  const result = await checkCapturePrecondition('macOS');
  assert.equal(result.ok, false);
  assert.equal(result.code, 'capture_precondition_unsupported_surface');
  assert.match(result.message, /--surface must be/);
}

// Happy-path assertions on whichever surface actually works on this host — we
// do not assume either, we only assert the shape when ok=true.
{
  for (const surface of ['node', 'browser']) {
    const result = await checkCapturePrecondition(surface);
    if (result.ok) {
      assert.equal(result.surface, surface);
      // Happy-path must not carry a blocker code.
      assert.equal(result.code, undefined);
    } else {
      assert.equal(result.surface, surface);
      assert.equal(typeof result.code, 'string');
      assert.ok(result.code.startsWith('capture_precondition_'));
      assert.equal(typeof result.message, 'string');
    }
  }
}

console.log('doppler-bundle-capture-precondition.test: ok');
