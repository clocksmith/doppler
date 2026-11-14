export interface NodeRuntimeModules {
  harness: typeof import('./browser-harness.js');
  runtime: typeof import('../config/runtime.js');
}

export interface NodeRuntimeBridge {
  loadRuntimeConfigFromRef: typeof import('./browser-harness/runtime-config.js').loadRuntimeConfigFromRef;
  loadRuntimeProfile: NodeRuntimeModules['harness']['loadRuntimeProfile'];
  loadRuntimeConfigFromUrl: NodeRuntimeModules['harness']['loadRuntimeConfigFromUrl'];
  getRuntimeConfig: NodeRuntimeModules['runtime']['getRuntimeConfig'];
  setRuntimeConfig: NodeRuntimeModules['runtime']['setRuntimeConfig'];
  resetRuntimeConfig: NodeRuntimeModules['runtime']['resetRuntimeConfig'];
  getActiveKernelPath: typeof import('../config/kernel-path-loader.js').getActiveKernelPath;
  getActiveKernelPathPolicy: typeof import('../config/kernel-path-loader.js').getActiveKernelPathPolicy;
  getActiveKernelPathSource: typeof import('../config/kernel-path-loader.js').getActiveKernelPathSource;
  setActiveKernelPath: typeof import('../config/kernel-path-loader.js').setActiveKernelPath;
}

export declare function loadRuntimeModules(): Promise<NodeRuntimeModules>;
export declare function createRuntimeBridge(modules: NodeRuntimeModules): NodeRuntimeBridge;
