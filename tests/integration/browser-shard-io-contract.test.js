import assert from 'node:assert/strict';

import { BrowserShardIO } from '../../src/experimental/browser/shard-io-browser.js';
import { setOpfsPathConfig } from '../../src/storage/shard-manager.js';

const originalNavigator = globalThis.navigator;

function createDirectoryHandle(label) {
  return {
    label,
    async getDirectoryHandle(name) {
      return createDirectoryHandle(`${label}/${name}`);
    },
  };
}

try {
  const root = createDirectoryHandle('root');
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      storage: {
        async getDirectory() {
          return root;
        },
      },
    },
    configurable: true,
  });

  setOpfsPathConfig({ opfsRootDir: 'alt-model-root' });
  const shardIo = await BrowserShardIO.create('demo-model');
  assert.equal(shardIo.getModelDir().label, 'root/alt-model-root/demo-model');
} finally {
  setOpfsPathConfig(null);
  if (originalNavigator === undefined) {
    delete globalThis.navigator;
  } else {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  }
}

console.log('browser-shard-io-contract.test: ok');
