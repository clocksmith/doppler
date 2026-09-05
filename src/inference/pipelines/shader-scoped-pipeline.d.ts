import type { ShaderSourceScope } from '../../gpu/kernels/shader-source-scope.js';
export declare function scopePipelineShaders<T extends object>(pipeline: T, scope: ShaderSourceScope | null): T;
