

import { MultiModelNetwork } from './multi-model-network.js';
import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

// Re-export primitives from multi-model-network
export { MultiModelNetwork };



// ============================================================================
// Primitive Types (for Reploid to use)
// ============================================================================















// ============================================================================
// Deprecation Notice
// ============================================================================


export class FunctionGemma {
  
  constructor(pipeline, loader, pool, config = {}) {
    log.warn(
      'FunctionGemma',
      '[DEPRECATED] FunctionGemma class is deprecated. ' +
      'Use Reploid\'s FunctionGemmaOrchestrator for orchestration, ' +
      'or MultiModelNetwork directly for primitives.'
    );

    
    this.network = new MultiModelNetwork(pipeline, loader, pool);

    
    this.pipeline = pipeline;

    
    const runtime = getRuntimeConfig();
    this.config = {
      defaultTemperature: runtime.inference.sampling.temperature,
      defaultMaxTokens: runtime.inference.batching.maxTokens,
      ...config,
    };
  }

  
  registerExpert(expert) {
    this.network.registerExpert(expert);
  }

  
  getExpert(id) {
    return this.network.getExpert(id);
  }

  
  listExperts() {
    return this.network.listExperts();
  }

  
  async executeExpert(expertId, prompt, options = {}) {
    return this.network.executeExpert(expertId, prompt, {
      maxTokens: this.config.defaultMaxTokens,
      temperature: this.config.defaultTemperature,
      ...options,
    });
  }

  
  async setSharedPrefix(prompt, options = {}) {
    return this.network.setSharedPrefix(prompt, options);
  }

  
  getNetwork() {
    return this.network;
  }

  
  getPipeline() {
    return this.pipeline;
  }
}
