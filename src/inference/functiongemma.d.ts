/**
 * FunctionGemma Primitives
 *
 * Doppler provides execution primitives only. Orchestration lives in Reploid.
 *
 * @module inference/functiongemma
 * @see reploid/src/capabilities/intelligence/functiongemma-orchestrator.js
 */
export {
  MultiModelNetwork,
  type ExpertNode,
  type CombinerConfig,
  type ExpertTask,
  type TopologyRouter,
} from './multi-model-network.js';
