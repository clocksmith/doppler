#!/usr/bin/env node
// Compatibility command for Doppler Rig.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { main } from './rig-model-capsule.js';
export { FORGE_VERSION, forgeModelCapsule } from '../src/tooling/model-capsule-forge.js';
export {
  buildRigOptions as buildForgeOptions,
  parseArgs, readJsonInput, usage, main,
} from './rig-model-capsule.js';

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-rig] ${error.message}`);
    process.exit(1);
  });
}
