import assert from 'node:assert/strict';

import {
  summarizeWgslAuthorBrowserExecution,
} from '../../tools/lib/wgsl-author-browser-executor.js';

const plan = {
  schema: 'doppler.wgsl-author-execution-plan/v1',
  modules: [
    { id: 'compute-module', wgsl: '@compute @workgroup_size(1) fn main() {}' },
    {
      id: 'render-module',
      wgsl: '@vertex fn v() -> @builtin(position) vec4f { return vec4f(); }\n'
        + '@fragment fn f() -> @location(0) vec4f { return vec4f(); }',
    },
  ],
  passes: [
    { id: 'prepare', kind: 'compute' },
    { id: 'draw', kind: 'render' },
  ],
  outputs: ['color'],
};
const deviceInfo = {
  vendor: 'test',
  availableFeatures: [],
  requiredFeatures: [],
  limits: {},
  requiredLimits: {},
};
const browserArgs = ['--test'];
const successful = summarizeWgslAuthorBrowserExecution(
  'receipt-test',
  plan,
  deviceInfo,
  browserArgs,
  {
    compilation: [
      { moduleId: 'compute-module', messages: [] },
      { moduleId: 'render-module', messages: [] },
    ],
    runtimeErrors: [],
    executedPassIds: ['prepare', 'draw'],
    outputs: { color: { kind: 'texture', bytes: [0, 1, 2, 3] } },
  }
);
assert.equal(successful.passed, true);
assert.equal(successful.validationErrorsAbsent, true);
assert.match(successful.planSha256, /^[a-f0-9]{64}$/);
assert.match(successful.receiptHash, /^[a-f0-9]{64}$/);
assert.deepEqual(
  summarizeWgslAuthorBrowserExecution(
    'receipt-test',
    plan,
    deviceInfo,
    browserArgs,
    {
      compilation: [
        { moduleId: 'compute-module', messages: [] },
        { moduleId: 'render-module', messages: [] },
      ],
      runtimeErrors: [],
      executedPassIds: ['prepare', 'draw'],
      outputs: { color: { kind: 'texture', bytes: [0, 1, 2, 3] } },
    }
  ),
  successful
);

const compilationFailure = summarizeWgslAuthorBrowserExecution(
  'compile-failure',
  plan,
  deviceInfo,
  browserArgs,
  {
    compilation: [
      {
        moduleId: 'compute-module',
        messages: [{ type: 'error', message: 'bad shader' }],
      },
      { moduleId: 'render-module', messages: [] },
    ],
    runtimeErrors: [],
    executedPassIds: [],
    outputs: {},
  }
);
assert.equal(compilationFailure.passed, false);
assert.equal(compilationFailure.compilation[0].errorCount, 1);

const runtimeFailure = summarizeWgslAuthorBrowserExecution(
  'runtime-failure',
  plan,
  deviceInfo,
  browserArgs,
  {
    compilation: [
      { moduleId: 'compute-module', messages: [] },
      { moduleId: 'render-module', messages: [] },
    ],
    runtimeErrors: ['validation failed'],
    executedPassIds: ['prepare'],
    outputs: {},
  }
);
assert.equal(runtimeFailure.passed, false);
assert.equal(runtimeFailure.validationErrorsAbsent, false);

console.log('wgsl-author-browser-executor.test: ok');
