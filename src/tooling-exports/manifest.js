// Narrow manifest/format export for consumers that need parseManifest +
// manifest introspection without pulling storage or device modules.

export {
  parseManifest,
  getManifest,
  setManifest,
  clearManifest,
  classifyTensorRole,
} from '../formats/rdrr/index.js';
export { DEFAULT_MANIFEST_INFERENCE } from '../config/schema/index.js';
