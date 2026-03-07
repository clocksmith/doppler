import assert from 'node:assert/strict';

import {
  Command,
  decodeHeader,
  encodeMessage,
} from '../../src/bridge/protocol.js';
import {
  createBridgeClient,
  getBridgeClient,
  readFileNative,
} from '../../src/bridge/index.js';

function createEvent() {
  const listeners = [];
  return {
    addListener(listener) {
      listeners.push(listener);
    },
    emit(value) {
      for (const listener of listeners) {
        listener(value);
      }
    },
  };
}

function createChromeRuntime(onPortMessage) {
  return {
    runtime: {
      connect(...args) {
        const extensionId = args.length === 2 ? args[0] : null;
        const onMessage = createEvent();
        const onDisconnect = createEvent();
        return {
          name: 'doppler-bridge',
          onMessage,
          onDisconnect,
          postMessage(message) {
            onPortMessage({ extensionId, message, onMessage, onDisconnect });
          },
          disconnect() {
            onDisconnect.emit();
          },
        };
      },
    },
  };
}

const originalChrome = globalThis.chrome;

try {
  globalThis.chrome = createChromeRuntime(({ extensionId, message, onMessage }) => {
    assert.equal(extensionId, 'ext-1');
    const payload = new Uint8Array(message.data);
    const header = decodeHeader(payload.buffer);
    assert.ok(header);
    queueMicrotask(() => {
      onMessage.emit({
        type: 'binary',
        data: Array.from(new Uint8Array(encodeMessage(Command.PONG, header.reqId))),
      });
    });
  });

  const client = await createBridgeClient('ext-1');
  assert.equal(client.isConnected(), true);
  assert.equal(client.getExtensionId(), 'ext-1');

  await assert.rejects(
    () => readFileNative('/tmp/model.bin', 0, 1, 'ext-2'),
    /different extension target/
  );

  client.disconnect();
} finally {
  getBridgeClient().disconnect();
  if (originalChrome === undefined) {
    delete globalThis.chrome;
  } else {
    globalThis.chrome = originalChrome;
  }
}

try {
  globalThis.chrome = createChromeRuntime(({ onMessage }) => {
    queueMicrotask(() => {
      onMessage.emit({
        type: 'error',
        message: 'native unavailable',
      });
    });
  });

  await assert.rejects(
    () => createBridgeClient('ext-error'),
    /native unavailable/
  );
} finally {
  getBridgeClient().disconnect();
  if (originalChrome === undefined) {
    delete globalThis.chrome;
  } else {
    globalThis.chrome = originalChrome;
  }
}

console.log('bridge-extension-client-contract.test: ok');
