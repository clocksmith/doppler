export function causalConvSiluForward(input: unknown, weight: unknown, options: unknown): {
    output: Float32Array<ArrayBufferLike>;
    cache: {
        raw: Float32Array<ArrayBufferLike>;
        numTokens: unknown;
        channels: unknown;
        kernelSize: unknown;
    };
};
export function causalConvSiluBackward(input: unknown, weight: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
    input: Float32Array<ArrayBufferLike>;
    weight: Float32Array<ArrayBufferLike>;
};
export function gatedRmsNormForward(input: unknown, gate: unknown, weight: unknown, options: unknown): {
    output: Float32Array<ArrayBufferLike>;
    cache: {
        inverseRms: Float32Array<ArrayBufferLike>;
        rows: unknown;
        width: unknown;
        eps: number;
    };
};
export function gatedRmsNormBackward(input: unknown, gate: unknown, weight: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
    input: Float32Array<ArrayBufferLike>;
    gate: Float32Array<ArrayBufferLike>;
    weight: Float32Array<ArrayBufferLike>;
};
export function l2NormalizeForward(input: unknown, options: unknown): {
    output: Float32Array<ArrayBufferLike>;
    cache: {
        inverseNorm: Float32Array<ArrayBufferLike>;
        rows: unknown;
        width: unknown;
        eps: number;
    };
};
export function l2NormalizeBackward(input: unknown, gradOutput: unknown, cache: unknown, options: unknown): Float32Array<ArrayBufferLike>;
export function qwenLinearAttentionPrepareForward(inputs: unknown, options: unknown): {
    query: Float32Array<ArrayBuffer>;
    key: Float32Array<ArrayBuffer>;
    value: Float32Array<ArrayBuffer>;
    logDecay: Float32Array<ArrayBuffer>;
    beta: Float32Array<ArrayBuffer>;
    cache: {
        dims: {
            numTokens: unknown;
            numKeyHeads: unknown;
            numValueHeads: unknown;
            keyDim: unknown;
            valueDim: unknown;
            eps: number;
            repeatFactor: number;
            querySize: number;
            keySize: number;
            valueSize: number;
            convSize: number;
        };
    };
};
export function qwenLinearAttentionPrepareBackward(inputs: unknown, gradients: unknown, cache: unknown, options: unknown): {
    mixed: Float32Array<ArrayBuffer>;
    a: Float32Array<ArrayBuffer>;
    b: Float32Array<ArrayBuffer>;
    aLog: Float32Array<ArrayBuffer>;
    dtBias: Float32Array<ArrayBuffer>;
};
export function gatedDeltaParametersForward(a: unknown, b: unknown, aLog: unknown, dtBias: unknown): {
    logDecay: Float32Array<ArrayBuffer>;
    beta: Float32Array<ArrayBuffer>;
};
export function gatedDeltaParametersBackward(a: unknown, b: unknown, aLog: unknown, dtBias: unknown, gradLogDecay: unknown, gradBeta: unknown): {
    a: Float32Array<ArrayBufferLike>;
    b: Float32Array<ArrayBufferLike>;
    aLog: Float32Array<ArrayBufferLike>;
    dtBias: Float32Array<ArrayBufferLike>;
};
export function qwenLinearAttentionCoreForward(inputs: unknown, options: unknown): {
    output: Float32Array<ArrayBufferLike>;
    finalState: Float32Array<ArrayBuffer>;
    cache: {
        dims: {
            kernelSize: unknown;
            checkpointInterval: unknown;
            queryScale: number;
            rmsEps: number;
            numTokens: unknown;
            numKeyHeads: unknown;
            numValueHeads: unknown;
            keyDim: unknown;
            valueDim: unknown;
            eps: number;
            repeatFactor: number;
            querySize: number;
            keySize: number;
            valueSize: number;
            convSize: number;
        };
        convolution: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                raw: Float32Array<ArrayBufferLike>;
                numTokens: unknown;
                channels: unknown;
                kernelSize: unknown;
            };
        };
        preparation: {
            query: Float32Array<ArrayBuffer>;
            key: Float32Array<ArrayBuffer>;
            value: Float32Array<ArrayBuffer>;
            logDecay: Float32Array<ArrayBuffer>;
            beta: Float32Array<ArrayBuffer>;
            cache: {
                dims: {
                    numTokens: unknown;
                    numKeyHeads: unknown;
                    numValueHeads: unknown;
                    keyDim: unknown;
                    valueDim: unknown;
                    eps: number;
                    repeatFactor: number;
                    querySize: number;
                    keySize: number;
                    valueSize: number;
                    convSize: number;
                };
            };
        };
        recurrence: {
            output: Float32Array<ArrayBuffer>;
            finalState: Float32Array<ArrayBuffer>;
            cache: {
                checkpoints: Float32Array<ArrayBuffer>;
                checkpointTokens: Uint32Array<ArrayBuffer>;
                checkpointInterval: number;
                blockCount: number;
                dims: {
                    numTokens: unknown;
                    numHeads: unknown;
                    keyDim: unknown;
                    valueDim: unknown;
                    queryScale: number;
                };
            };
        };
        normalization: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                inverseRms: Float32Array<ArrayBufferLike>;
                rows: unknown;
                width: unknown;
                eps: number;
            };
        };
    };
};
export function qwenLinearAttentionCoreBackward(inputs: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
    qkv: Float32Array<ArrayBufferLike>;
    z: Float32Array<ArrayBufferLike>;
    a: Float32Array<ArrayBuffer>;
    b: Float32Array<ArrayBuffer>;
    initialState: null;
    convWeight: Float32Array<ArrayBufferLike>;
    normWeight: Float32Array<ArrayBufferLike>;
    aLog: Float32Array<ArrayBuffer>;
    dtBias: Float32Array<ArrayBuffer>;
};
export function qwenLinearAttentionModuleForward(inputs: unknown, options: unknown): {
    output: Float32Array<ArrayBuffer>;
    finalState: Float32Array<ArrayBuffer>;
    cache: {
        dims: {
            hiddenSize: unknown;
            kernelSize: unknown;
            checkpointInterval: unknown;
            queryScale: number;
            rmsEps: number;
            numTokens: unknown;
            numKeyHeads: unknown;
            numValueHeads: unknown;
            keyDim: unknown;
            valueDim: unknown;
            eps: number;
            repeatFactor: number;
            querySize: number;
            keySize: number;
            valueSize: number;
            convSize: number;
        };
        projections: {
            qkv: Float32Array<ArrayBuffer>;
            z: Float32Array<ArrayBuffer>;
            a: Float32Array<ArrayBuffer>;
            b: Float32Array<ArrayBuffer>;
        };
        core: {
            output: Float32Array<ArrayBufferLike>;
            finalState: Float32Array<ArrayBuffer>;
            cache: {
                dims: {
                    kernelSize: unknown;
                    checkpointInterval: unknown;
                    queryScale: number;
                    rmsEps: number;
                    numTokens: unknown;
                    numKeyHeads: unknown;
                    numValueHeads: unknown;
                    keyDim: unknown;
                    valueDim: unknown;
                    eps: number;
                    repeatFactor: number;
                    querySize: number;
                    keySize: number;
                    valueSize: number;
                    convSize: number;
                };
                convolution: {
                    output: Float32Array<ArrayBufferLike>;
                    cache: {
                        raw: Float32Array<ArrayBufferLike>;
                        numTokens: unknown;
                        channels: unknown;
                        kernelSize: unknown;
                    };
                };
                preparation: {
                    query: Float32Array<ArrayBuffer>;
                    key: Float32Array<ArrayBuffer>;
                    value: Float32Array<ArrayBuffer>;
                    logDecay: Float32Array<ArrayBuffer>;
                    beta: Float32Array<ArrayBuffer>;
                    cache: {
                        dims: {
                            numTokens: unknown;
                            numKeyHeads: unknown;
                            numValueHeads: unknown;
                            keyDim: unknown;
                            valueDim: unknown;
                            eps: number;
                            repeatFactor: number;
                            querySize: number;
                            keySize: number;
                            valueSize: number;
                            convSize: number;
                        };
                    };
                };
                recurrence: {
                    output: Float32Array<ArrayBuffer>;
                    finalState: Float32Array<ArrayBuffer>;
                    cache: {
                        checkpoints: Float32Array<ArrayBuffer>;
                        checkpointTokens: Uint32Array<ArrayBuffer>;
                        checkpointInterval: number;
                        blockCount: number;
                        dims: {
                            numTokens: unknown;
                            numHeads: unknown;
                            keyDim: unknown;
                            valueDim: unknown;
                            queryScale: number;
                        };
                    };
                };
                normalization: {
                    output: Float32Array<ArrayBufferLike>;
                    cache: {
                        inverseRms: Float32Array<ArrayBufferLike>;
                        rows: unknown;
                        width: unknown;
                        eps: number;
                    };
                };
            };
        };
    };
};
export function qwenLinearAttentionModuleBackward(inputs: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
    hidden: Float32Array<ArrayBuffer>;
    initialState: null;
};
