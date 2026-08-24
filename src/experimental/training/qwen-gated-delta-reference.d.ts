export function gatedDeltaRecurrentForward(inputs: any, options: any): {
    output: Float32Array<ArrayBuffer>;
    finalState: Float32Array<ArrayBuffer>;
    cache: {
        states: Float32Array<ArrayBuffer>;
        dims: {
            numTokens: any;
            numHeads: any;
            keyDim: any;
            valueDim: any;
            queryScale: number;
        };
    };
};
export function gatedDeltaRecurrentBackward(inputs: any, gradOutput: any, cache: any, options: any): {
    query: Float32Array<ArrayBuffer>;
    key: Float32Array<ArrayBuffer>;
    value: Float32Array<ArrayBuffer>;
    logDecay: Float32Array<ArrayBuffer>;
    beta: Float32Array<ArrayBuffer>;
    initialState: Float32Array<ArrayBuffer>;
};
export function estimateGatedDeltaCheckpointElements(options: any): {
    checkpointInterval: number;
    blockCount: number;
    stateElements: number;
    fullHistoryElements: number;
    storedCheckpointElements: number;
    peakRecomputedBlockElements: number;
    peakBackwardStateElements: number;
};
export function gatedDeltaRecurrentCheckpointedForward(inputs: any, options: any): {
    output: Float32Array<ArrayBuffer>;
    finalState: Float32Array<ArrayBuffer>;
    cache: {
        checkpoints: Float32Array<ArrayBuffer>;
        checkpointTokens: Uint32Array<ArrayBuffer>;
        checkpointInterval: number;
        blockCount: number;
        dims: {
            numTokens: any;
            numHeads: any;
            keyDim: any;
            valueDim: any;
            queryScale: number;
        };
    };
};
export function gatedDeltaRecurrentCheckpointedBackward(inputs: any, gradOutput: any, cache: any, options: any): {
    query: Float32Array<any>;
    key: Float32Array<any>;
    value: Float32Array<any>;
    logDecay: Float32Array<any>;
    beta: Float32Array<any>;
    initialState: null;
};
