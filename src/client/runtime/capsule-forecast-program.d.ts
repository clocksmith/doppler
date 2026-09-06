import type { RuntimePorts } from './composition-root.js';
export declare function createForecastProgramFactory(device: GPUDevice | { getDevice(): GPUDevice }): RuntimePorts['programFactory'];
