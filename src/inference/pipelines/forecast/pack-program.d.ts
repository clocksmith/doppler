import type { RuntimePorts } from '../../../client/runtime/composition-root.js';
export declare function createForecastProgramFactory(device: GPUDevice | { getDevice(): GPUDevice }): RuntimePorts['programFactory'];
