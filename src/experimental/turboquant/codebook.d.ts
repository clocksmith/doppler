export function retainTurboQuantSharedBuffers(device: any, options?: {}): {
    rotationMatrixBuffer: any;
    codebookCentroidsBuffer: any;
    codebookBoundariesBuffer: any;
    qjlMatrixBuffer: any;
    release(): void;
};
/**
 * Compute packed stride for a given headDim and bit-width.
 * packedStride = ceil(headDim / packFactor) where packFactor = floor(32 / bitWidth).
 *
 * @param {number} headDim
 * @param {number} bitWidth
 * @returns {number}
 */
export function computePackedStride(headDim: number, bitWidth: number): number;
