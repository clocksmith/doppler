export function causalConvSiluForward(input: any, weight: any, options: any): {
    output: Float32Array<any>;
    cache: {
        raw: Float32Array<any>;
        numTokens: any;
        channels: any;
        kernelSize: any;
    };
};
export function causalConvSiluBackward(input: any, weight: any, gradOutput: any, cache: any, options: any): {
    input: Float32Array<any>;
    weight: Float32Array<any>;
};
export function gatedRmsNormForward(input: any, gate: any, weight: any, options: any): {
    output: Float32Array<any>;
    cache: {
        inverseRms: Float32Array<any>;
        rows: any;
        width: any;
        eps: number;
    };
};
export function gatedRmsNormBackward(input: any, gate: any, weight: any, gradOutput: any, cache: any, options: any): {
    input: Float32Array<any>;
    gate: Float32Array<any>;
    weight: Float32Array<any>;
};
export function l2NormalizeForward(input: any, options: any): {
    output: Float32Array<any>;
    cache: {
        inverseNorm: Float32Array<any>;
        rows: any;
        width: any;
        eps: number;
    };
};
export function l2NormalizeBackward(input: any, gradOutput: any, cache: any, options: any): Float32Array<any>;
export function gatedDeltaParametersForward(a: any, b: any, aLog: any, dtBias: any): {
    logDecay: Float32Array<ArrayBuffer>;
    beta: Float32Array<ArrayBuffer>;
};
export function gatedDeltaParametersBackward(a: any, b: any, aLog: any, dtBias: any, gradLogDecay: any, gradBeta: any): {
    a: Float32Array<any>;
    b: Float32Array<any>;
    aLog: Float32Array<any>;
    dtBias: Float32Array<any>;
};
