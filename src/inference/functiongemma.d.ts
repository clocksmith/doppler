/**
 * FunctionGemma Primitives
 *
 * Doppler provides execution primitives only. Orchestration lives in the host app.
 *
 * @module inference/functiongemma
 * @see ../../docs/ARCHITECTURE.md#engine-vs-orchestrator-boundary
 */
export {
  MultiModelNetwork,
  type ExpertNode,
  type CombinerConfig,
  type ExpertTask,
  type TopologyRouter,
} from './multi-model-network.js';
