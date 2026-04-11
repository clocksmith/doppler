import assert from 'node:assert/strict';

import { assertSupportedGenerationOptions } from '../../src/client/runtime/model-session.js';

assert.throws(() => assertSupportedGenerationOptions({ stopTokens: [1, 2] }), /do not support stopTokens/);

console.log('doppler-generation-contract.test: ok');
