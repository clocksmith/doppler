import assert from 'node:assert/strict';
import path from 'node:path';
import { tmpdir } from 'node:os';

import { runBrowserCommandInNode } from '../../src/tooling/node-browser-command-runner.js';

const KERNELS_REQUEST = {
  command: 'test-model',
  suite: 'kernels',
};

await assert.rejects(
  () => runBrowserCommandInNode({ ...KERNELS_REQUEST, keepPipeline: true }),
  /browser command relay does not support keepPipeline=true/
);

await assert.rejects(
  () => runBrowserCommandInNode({
    command: 'convert',
    inputDir: '/tmp/input',
    convertPayload: {
      converterConfig: {},
    },
  }),
  /browser command relay does not support convert/
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'not-an-absolute-url',
  }),
  /browser command: baseUrl must be an absolute URL/
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    headless: 'maybe',
  }),
  /browser command: headless must be true or false\./
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    timeoutMs: 0,
  }),
  /browser command: timeoutMs must be a positive number\./
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    browserArgs: '--disable-gpu',
  }),
  /browser command: browserArgs must be an array\./
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    browserArgs: ['--disable-gpu', 7],
  }),
  /browser command: --browser-arg values must be strings\./
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    browserArgs: [null],
  }),
  /browser command: --browser-arg values must be strings\./
);

await assert.rejects(
  () => runBrowserCommandInNode({
    ...KERNELS_REQUEST,
    loadMode: 'opfs',
  }, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: false,
  }),
  /browser command: loadMode=opfs requires OPFS cache support/
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: false,
    timeoutMs: 1500,
    executablePath: '/definitely/missing/chrome',
  }),
  /browser command: failed to launch browser/
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: true,
    userDataDir: path.join(tmpdir(), `doppler-opfs-cache-${Date.now()}`),
    wipeCacheBeforeLaunch: true,
    timeoutMs: 1500,
    executablePath: '/definitely/missing/chrome',
  }),
  /browser command: failed to launch persistent browser/
);

await assert.rejects(
  () => runBrowserCommandInNode({
    ...KERNELS_REQUEST,
    loadMode: 'opfs',
  }, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: true,
    timeoutMs: 1500,
  }),
  /browser command: loadMode=opfs requires persistent browser context; persistent launch failed\./
);

{
  const warnings = [];
  await assert.rejects(
    () => runBrowserCommandInNode({
      ...KERNELS_REQUEST,
      loadMode: 'opfs',
    }, {
      baseUrl: 'http://127.0.0.1:1',
      opfsCache: true,
      timeoutMs: 1500,
      onConsole(message) {
        warnings.push(message);
      },
    }),
    /browser command: loadMode=opfs requires persistent browser context; persistent launch failed\./
  );

  assert.ok(
    warnings.some((message) => message?.text?.includes('Persistent browser launch failed')),
    'expected persistent-launch warning to be emitted'
  );
}

{
  const warnings = [];
  await assert.rejects(
    () => runBrowserCommandInNode({
      ...KERNELS_REQUEST,
      modelUrl: 'https://example.com/model/',
    }, {
      baseUrl: 'http://127.0.0.1:1',
      opfsCache: true,
      timeoutMs: 1500,
      onConsole(message) {
        warnings.push(message);
      },
    }),
    /browser command: failed to launch browser/
  );

  assert.ok(
    warnings.some((message) => message?.text?.includes('Persistent launch still failing; falling back to non-persistent mode.')),
    'expected fallback-to-non-persistent warning to be emitted'
  );
}

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: false,
    timeoutMs: 1500,
  }),
  /browser command: failed to launch browser/
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    runnerPath: 'src/tooling/command-runner.html',
    opfsCache: false,
    timeoutMs: 1500,
    executablePath: '/definitely/missing/chrome',
  }),
  /browser command: failed to launch browser/
);

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    opfsCache: false,
    timeoutMs: 1500,
  }),
  /browser command: failed to start static server/
);

console.log('node-browser-command-relay-contract.test: ok');
