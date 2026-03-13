import assert from 'node:assert/strict';

const {
  resolveRuntimeFromConfig,
  applyOrderedRuntimeInputs,
} = await import('../../src/tooling/runtime-input-composition.js');

// === resolveRuntimeFromConfig ===

assert.equal(resolveRuntimeFromConfig(null), null);
assert.equal(resolveRuntimeFromConfig(undefined), null);
assert.equal(resolveRuntimeFromConfig('string'), null);

{
  const runtime = { shared: { debug: {} } };
  const config = { runtime };
  assert.equal(resolveRuntimeFromConfig(config), runtime);
}

{
  const flat = { shared: { debug: {} }, inference: { prompt: 'hi' } };
  assert.equal(resolveRuntimeFromConfig(flat), flat);
}

assert.equal(resolveRuntimeFromConfig({ foo: 'bar' }), null);

// === applyOrderedRuntimeInputs ===

function createRuntimeBridge() {
  let config = {};
  return {
    getRuntimeConfig() { return config; },
    setRuntimeConfig(c) { config = c; },
    get current() { return config; },
  };
}

// Missing bridge methods
await assert.rejects(
  () => applyOrderedRuntimeInputs(null, {}),
  /runtime bridge must provide setRuntimeConfig/
);

await assert.rejects(
  () => applyOrderedRuntimeInputs({ setRuntimeConfig() {} }, {}),
  /runtime bridge must provide getRuntimeConfig/
);

// Empty inputs — no-op
{
  const bridge = createRuntimeBridge();
  await applyOrderedRuntimeInputs(bridge, {});
  assert.deepEqual(bridge.current, {});
}

// runtimeConfig applies
{
  const bridge = createRuntimeBridge();
  await applyOrderedRuntimeInputs(bridge, {
    runtimeConfig: { inference: { prompt: 'test' } },
  });
  assert.equal(bridge.current.inference?.prompt, 'test');
}

// runtimeConfig with bad shape throws
await assert.rejects(
  () => applyOrderedRuntimeInputs(createRuntimeBridge(), {
    runtimeConfig: { notARuntime: true },
  }),
  /runtimeConfig is missing runtime fields/
);

// configChain without handler throws
await assert.rejects(
  () => applyOrderedRuntimeInputs(createRuntimeBridge(), {
    configChain: ['debug'],
  }),
  /does not support configChain/
);

// configChain applies in order
{
  const bridge = createRuntimeBridge();
  const loaded = [];
  await applyOrderedRuntimeInputs(bridge, {
    configChain: ['first', 'second'],
  }, {
    loadRuntimeConfigFromRef: async (ref) => {
      loaded.push(ref);
      return { inference: { prompt: ref } };
    },
  });
  assert.deepEqual(loaded, ['first', 'second']);
  assert.equal(bridge.current.inference?.prompt, 'second');
}

// runtimePreset without handler throws
await assert.rejects(
  () => applyOrderedRuntimeInputs(createRuntimeBridge(), {
    runtimePreset: 'debug',
  }),
  /does not support runtimePreset/
);

// runtimePreset applies
{
  const bridge = createRuntimeBridge();
  let presetApplied = null;
  await applyOrderedRuntimeInputs(bridge, {
    runtimePreset: 'debug',
  }, {
    applyRuntimePreset: async (name) => { presetApplied = name; },
  });
  assert.equal(presetApplied, 'debug');
}

// runtimeConfigUrl without handler throws
await assert.rejects(
  () => applyOrderedRuntimeInputs(createRuntimeBridge(), {
    runtimeConfigUrl: 'https://example.com/config.json',
  }),
  /does not support runtimeConfigUrl/
);

// runtimeContractPatch as function
{
  const bridge = createRuntimeBridge();
  await applyOrderedRuntimeInputs(bridge, {
    runtimeContractPatch: () => ({ shared: { tooling: { intent: 'calibrate' } } }),
  });
  assert.equal(bridge.current.shared?.tooling?.intent, 'calibrate');
}

// runtimeContractPatch as object
{
  const bridge = createRuntimeBridge();
  await applyOrderedRuntimeInputs(bridge, {
    runtimeContractPatch: { shared: { tooling: { intent: 'investigate' } } },
  });
  assert.equal(bridge.current.shared?.tooling?.intent, 'investigate');
}

// Order: configChain -> runtimePreset -> runtimeConfigUrl -> runtimeConfig -> runtimeContractPatch
{
  const order = [];
  const bridge = createRuntimeBridge();
  await applyOrderedRuntimeInputs(bridge, {
    configChain: ['chain1'],
    runtimePreset: 'debug',
    runtimeConfigUrl: 'https://example.com',
    runtimeConfig: { inference: { prompt: 'test' } },
    runtimeContractPatch: { shared: { tooling: { intent: 'calibrate' } } },
  }, {
    loadRuntimeConfigFromRef: async () => { order.push('configChain'); return { inference: {} }; },
    applyRuntimePreset: async () => { order.push('runtimePreset'); },
    applyRuntimeConfigFromUrl: async () => { order.push('runtimeConfigUrl'); },
  });
  assert.deepEqual(order, ['configChain', 'runtimePreset', 'runtimeConfigUrl']);
}

console.log('runtime-input-composition.test: ok');
