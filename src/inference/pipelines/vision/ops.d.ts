/**
 * Layer norm on GPU.
 * @param {GPUBuffer} input   [seqLen, hiddenSize]
 * @param {GPUBuffer} weight  [hiddenSize]
 * @param {GPUBuffer} bias    [hiddenSize] or null
 * @param {{ seqLen: number, hiddenSize: number, eps: number }} opts
 * @returns {Promise<GPUBuffer>}
 */
export function doLayerNorm(input: GPUBuffer, weight: GPUBuffer, bias: GPUBuffer, opts: {
    seqLen: number;
    hiddenSize: number;
    eps: number;
}): Promise<GPUBuffer>;
/**
 * Matrix multiply on GPU.
 * @param {GPUBuffer} a  [M, K]
 * @param {GPUBuffer} b  [K, N]
 * @param {{ M: number, K: number, N: number, bias?: GPUBuffer }} opts
 * @returns {Promise<GPUBuffer>}
 */
export function doMatmul(a: GPUBuffer, b: GPUBuffer, opts: {
    M: number;
    K: number;
    N: number;
    bias?: GPUBuffer;
}): Promise<GPUBuffer>;
/**
 * GELU activation on GPU.
 * @param {GPUBuffer} input   Flat buffer
 * @param {{ count: number }} opts  Total element count
 * @returns {Promise<GPUBuffer>}
 */
export function doGelu(input: GPUBuffer, opts: {
    count: number;
}): Promise<GPUBuffer>;
/**
 * Element-wise residual add on GPU.
 * @param {GPUBuffer} a
 * @param {GPUBuffer} b
 * @param {{ count: number }} opts
 * @returns {Promise<GPUBuffer>}
 */
export function doResidualAdd(a: GPUBuffer, b: GPUBuffer, opts: {
    count: number;
}): Promise<GPUBuffer>;
