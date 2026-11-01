// Compatibility entry point; runtime observation state belongs to src/debug.
export {
  setDebugCategories,
  resetDebugConfig,
  applyPipelineDebugConfig,
  getDebugConfig,
  shouldDebugLayerOutput,
  isEnabled,
  formatTag,
  isBufferStatsEnabled,
  getMaxAbsThreshold,
} from '../../../../debug/pipeline-config.js';
export { incrementDecodeStep, resetDecodeStep, getDecodeStep } from '../../../../debug/config.js';
