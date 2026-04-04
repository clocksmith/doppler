import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';

import {
  finalizeBrowserRelayResponse,
  resolveLocalFileModelUrlForBrowserRelay,
  runBrowserCommandInNode,
} from '../../src/tooling/node-browser-command-runner.js';

const KERNELS_REQUEST = {
  command: 'verify',
  suite: 'kernels',
  runtimeConfig: {
    shared: {
      tooling: {
        intent: 'verify',
      },
    },
    inference: {
      kernelPathPolicy: {
        mode: 'capability-aware',
        sourceScope: ['model', 'manifest', 'config'],
        onIncompatible: 'remap',
      },
    },
  },
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

{
  const sourceRequest = {
    ...KERNELS_REQUEST,
    modelId: 'toy-model',
    modelUrl: 'https://example.com/model/',
  };
  const effectiveRequest = {
    ...sourceRequest,
    loadMode: 'opfs',
  };
  const response = finalizeBrowserRelayResponse({
    ok: true,
    schemaVersion: 1,
    surface: 'browser',
    request: {
      ...effectiveRequest,
    },
    result: {
      passed: 1,
      failed: 0,
    },
  }, effectiveRequest);
  assert.equal(response.request.loadMode, 'opfs');
  assert.equal(response.request.modelUrl, sourceRequest.modelUrl);
}

{
  const modelDir = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-relay-local-model-'));
  try {
    const resolution = await resolveLocalFileModelUrlForBrowserRelay({
      ...KERNELS_REQUEST,
      modelId: 'toy-local-model',
      modelUrl: pathToFileURL(modelDir).href,
    }, {
      staticMounts: [{
        urlPrefix: '/models/external',
        rootDir: '/tmp/external',
      }],
    });
    assert.ok(
      resolution.relayRequest.modelUrl.startsWith('/__doppler_local_model/'),
      `expected relay modelUrl mount, got ${resolution.relayRequest.modelUrl}`
    );
    assert.equal(resolution.staticMounts.length, 2);
    assert.equal(
      resolution.staticMounts[1].urlPrefix,
      resolution.relayRequest.modelUrl
    );
    assert.equal(
      resolution.staticMounts[1].rootDir,
      modelDir
    );
  } finally {
    await fs.rm(modelDir, { recursive: true, force: true });
  }
}

{
  const runtimeDir = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-relay-runtime-config-'));
  const runtimeConfigPath = path.join(runtimeDir, 'runtime-config.json');
  await fs.writeFile(runtimeConfigPath, JSON.stringify({
    extends: './base.json',
    runtime: {
      shared: {
        tooling: {
          intent: 'verify',
        },
      },
    },
  }), 'utf8');
  await fs.writeFile(path.join(runtimeDir, 'base.json'), JSON.stringify({
    runtime: {
      shared: {
        harness: {
          mode: 'verify',
        },
      },
    },
  }), 'utf8');
  try {
    const resolution = await resolveLocalFileModelUrlForBrowserRelay({
      ...KERNELS_REQUEST,
      runtimeConfigUrl: pathToFileURL(runtimeConfigPath).href,
    });
    assert.ok(
      resolution.relayRequest.runtimeConfigUrl.startsWith('/__doppler_local_runtime_config/'),
      `expected relay runtimeConfigUrl mount, got ${resolution.relayRequest.runtimeConfigUrl}`
    );
    assert.equal(resolution.staticMounts.length, 1);
    assert.equal(
      resolution.staticMounts[0].rootDir,
      runtimeDir
    );
    assert.ok(
      resolution.relayRequest.runtimeConfigUrl.endsWith('/runtime-config.json'),
      `expected runtime config relay URL to keep the filename, got ${resolution.relayRequest.runtimeConfigUrl}`
    );
  } finally {
    await fs.rm(runtimeDir, { recursive: true, force: true });
  }
}

{
  const imageDir = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-relay-input-image-'));
  const imagePath = path.join(imageDir, 'input.png');
  await fs.writeFile(imagePath, Buffer.from([137, 80, 78, 71]), 'binary');
  try {
    const resolution = await resolveLocalFileModelUrlForBrowserRelay({
      command: 'verify',
      workload: 'inference',
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      inferenceInput: {
        prompt: 'Describe the image.',
        image: {
          url: pathToFileURL(imagePath).href,
        },
      },
    });
    assert.ok(
      resolution.relayRequest.inferenceInput.image.url.startsWith('/__doppler_local_input_image/'),
      `expected relay image mount, got ${resolution.relayRequest.inferenceInput.image.url}`
    );
    assert.equal(resolution.staticMounts.length, 1);
    assert.equal(resolution.staticMounts[0].rootDir, imageDir);
    assert.ok(
      resolution.relayRequest.inferenceInput.image.url.endsWith('/input.png'),
      `expected image relay URL to keep the filename, got ${resolution.relayRequest.inferenceInput.image.url}`
    );
  } finally {
    await fs.rm(imageDir, { recursive: true, force: true });
  }
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

{
  const modelDir = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-relay-local-model-'));
  try {
    await assert.rejects(
      () => runBrowserCommandInNode({
        ...KERNELS_REQUEST,
        modelId: 'toy-local-model',
        modelUrl: pathToFileURL(modelDir).href,
      }, {
        baseUrl: 'http://127.0.0.1:1',
      }),
      /browser command: explicit local file:\/\/ modelUrl requires the relay-owned static server/
    );
  } finally {
    await fs.rm(modelDir, { recursive: true, force: true });
  }
}

{
  const runtimeDir = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-relay-runtime-config-'));
  const runtimeConfigPath = path.join(runtimeDir, 'runtime-config.json');
  await fs.writeFile(runtimeConfigPath, JSON.stringify({ runtime: {} }), 'utf8');
  try {
    await assert.rejects(
      () => runBrowserCommandInNode({
        ...KERNELS_REQUEST,
        runtimeConfigUrl: pathToFileURL(runtimeConfigPath).href,
      }, {
        baseUrl: 'http://127.0.0.1:1',
      }),
      /browser command: explicit local file:\/\/ runtimeConfigUrl requires the relay-owned static server/
    );
  } finally {
    await fs.rm(runtimeDir, { recursive: true, force: true });
  }
}

{
  const imageDir = await fs.mkdtemp(path.join(tmpdir(), 'doppler-browser-relay-input-image-'));
  const imagePath = path.join(imageDir, 'input.png');
  await fs.writeFile(imagePath, Buffer.from([137, 80, 78, 71]), 'binary');
  try {
    await assert.rejects(
      () => runBrowserCommandInNode({
        command: 'verify',
        workload: 'inference',
        modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
        inferenceInput: {
          prompt: 'Describe the image.',
          image: {
            url: pathToFileURL(imagePath).href,
          },
        },
      }, {
        baseUrl: 'http://127.0.0.1:1',
      }),
      /browser command: explicit local file:\/\/ inferenceInput\.image\.url requires the relay-owned static server/
    );
  } finally {
    await fs.rm(imageDir, { recursive: true, force: true });
  }
}

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
      /browser command: failed to start static server|browser command: failed to launch browser|ERR_UNSAFE_PORT|runner did not become ready|runtime\.inference\.kernelPathPolicy must not be null/
    );
  } else {
    assert.ok(result && typeof result === 'object');
    assert.equal(result.surface, 'browser');
    assert.equal(result.request?.command, 'verify');
  }
}

console.log('node-browser-command-relay-contract.test: ok');
