import assert from 'node:assert/strict';
import path from 'node:path';
import { tmpdir } from 'node:os';

import {
  finalizeBrowserRelayResponse,
  runBrowserCommandInNode,
} from '../../src/tooling/node-browser-command-runner.js';

const KERNELS_REQUEST = {
  command: 'verify',
  suite: 'kernels',
};

{
  const response = finalizeBrowserRelayResponse({
    ok: true,
    schemaVersion: 1,
    surface: 'browser',
    request: {
      ...KERNELS_REQUEST,
      loadMode: 'http',
    },
    result: {
      passed: 1,
      failed: 0,
    },
  }, KERNELS_REQUEST);
  assert.deepEqual(response.request, KERNELS_REQUEST);
}

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
    staticMounts: 'bad',
    opfsCache: false,
    timeoutMs: 1500,
  }),
  /browser command: staticMounts must be an array\./
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
    modelUrl: 'https://example.com/model',
  }, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: true,
  }),
  /browser command: loadMode=opfs requires modelId when modelUrl is provided/
);

await assert.rejects(
  () => runBrowserCommandInNode({
    ...KERNELS_REQUEST,
    loadMode: 'opfs',
    modelId: 'toy-model',
    modelUrl: 'https://example.com/model',
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
  /browser command: loadMode=opfs requires persistent browser context; persistent launch failed\.|ERR_UNSAFE_PORT/
);

{
  const warnings = [];
  let failedWithUnsafePort = false;
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
    (error) => {
      const message = String(error?.message || error);
      failedWithUnsafePort = message.includes('ERR_UNSAFE_PORT');
      return /browser command: loadMode=opfs requires persistent browser context; persistent launch failed\.|ERR_UNSAFE_PORT/.test(message);
    }
  );

  if (!failedWithUnsafePort) {
    assert.ok(
      warnings.some((message) => message?.text?.includes('Persistent browser launch failed')),
      'expected persistent-launch warning to be emitted'
    );
    assert.ok(
      warnings.every((message) => !message?.text?.includes('falling back to non-persistent mode')),
      'unexpected silent fallback-to-non-persistent warning'
    );
  }
}

{
  const warnings = [];
  let failedWithUnsafePort = false;
  await assert.rejects(
    () => runBrowserCommandInNode({
      ...KERNELS_REQUEST,
      modelId: 'toy-model',
      modelUrl: 'https://example.com/model/',
    }, {
      baseUrl: 'http://127.0.0.1:1',
      opfsCache: true,
      timeoutMs: 1500,
      onConsole(message) {
        warnings.push(message);
      },
    }),
    (error) => {
      const message = String(error?.message || error);
      failedWithUnsafePort = message.includes('ERR_UNSAFE_PORT');
      return /browser command: persistent browser context is required when OPFS cache is enabled; persistent launch failed\.|ERR_UNSAFE_PORT/.test(message);
    }
  );

  if (!failedWithUnsafePort) {
    assert.ok(
      warnings.some((message) => message?.text?.includes('Persistent browser launch failed')),
      'expected persistent-launch retry warning to be emitted'
    );
    assert.ok(
      warnings.every((message) => !message?.text?.includes('falling back to non-persistent mode')),
      'unexpected silent fallback-to-non-persistent warning'
    );
  }
}

await assert.rejects(
  () => runBrowserCommandInNode(KERNELS_REQUEST, {
    baseUrl: 'http://127.0.0.1:1',
    opfsCache: false,
    timeoutMs: 1500,
  }),
  /browser command: failed to launch browser|ERR_UNSAFE_PORT/
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

{
  let result = null;
  let error = null;
  try {
    result = await runBrowserCommandInNode(KERNELS_REQUEST, {
      opfsCache: false,
      timeoutMs: 1500,
    });
  } catch (nextError) {
    error = nextError;
  }

  if (error) {
    assert.match(
      String(error?.message || error),
      /browser command: failed to start static server|browser command: failed to launch browser|ERR_UNSAFE_PORT|runner did not become ready/
    );
  } else {
    assert.ok(result && typeof result === 'object');
    assert.equal(result.surface, 'browser');
    assert.equal(result.request?.command, 'verify');
  }
}

console.log('node-browser-command-relay-contract.test: ok');
