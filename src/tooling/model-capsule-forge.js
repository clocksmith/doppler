// Compatibility entrypoint for Doppler Rig.
export {
  RIG_VERSION as FORGE_VERSION,
  buildRigOptions as buildForgeOptions,
  rigModelCapsule as forgeModelCapsule,
  parseArgs, readJsonInput, usage,
} from './model-capsule-rig.js';
