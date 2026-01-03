

import { log } from '../debug/index.js';

let isWarmed = false;


export function markWarmed() {
  if (!isWarmed) {
    isWarmed = true;
    log.debug('KernelCache', 'Warmed');
  }
}
