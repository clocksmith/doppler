import { log } from '@simulatte/doppler';

import './demo-utils.js';
import './demo-routing.js';
import './demo-translate.js';
import './demo-storage.js';
import './demo-generation.js';
import './demo-diagnostics.js';
import { initDemo } from './demo-core.js';

export * from './demo-core.js';

initDemo().catch((error) => {
  log.error('DopplerDemo', `Demo init failed: ${error.message}`);
});
