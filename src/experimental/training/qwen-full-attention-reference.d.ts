export function qwenAttentionSplitQGateForward(input: unknown, options: unknown): {
    query: Float32Array<ArrayBuffer>;
    gate: Float32Array<ArrayBuffer>;
};
export function qwenAttentionSplitQGateBackward(gradQuery: unknown, gradGate: unknown, options: unknown): Float32Array<ArrayBuffer>;
export function sigmoidGateForward(input: unknown, gate: unknown): Float32Array<ArrayBuffer>;
export function sigmoidGateBackward(input: unknown, gate: unknown, gradOutput: unknown): {
    input: Float32Array<ArrayBufferLike>;
    gate: Float32Array<ArrayBufferLike>;
};
export function partialRopeForward(input: unknown, freqsCos: unknown, freqsSin: unknown, options: unknown): Float32Array<ArrayBufferLike>;
export function partialRopeBackward(gradOutput: unknown, freqsCos: unknown, freqsSin: unknown, options: unknown): Float32Array<ArrayBufferLike>;
export function qwenFullAttentionModuleForward(inputs: unknown, options: unknown): {
    output: Float32Array<ArrayBuffer>;
    cache: {
        dims: {
            numTokens: unknown;
            hiddenSize: unknown;
            numHeads: unknown;
            numKVHeads: unknown;
            headDim: unknown;
            rmsEps: number;
            querySize: number;
            kvSize: number;
            rotaryDim: unknown;
            pairSpanDim: unknown;
            startPos: number;
            interleaved: boolean;
            halfRotary: number;
        };
        split: {
            query: Float32Array<ArrayBuffer>;
            gate: Float32Array<ArrayBuffer>;
        };
        kProjection: Float32Array<ArrayBuffer>;
        value: Float32Array<ArrayBuffer>;
        queryNorm: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                inverseRms: Float32Array<ArrayBufferLike>;
            };
        };
        keyNorm: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                inverseRms: Float32Array<ArrayBufferLike>;
            };
        };
        query: Float32Array<ArrayBufferLike>;
        key: Float32Array<ArrayBufferLike>;
        attention: {
            output: Float32Array<ArrayBuffer>;
            softmax: Float32Array<ArrayBufferLike>;
        };
        gated: Float32Array<ArrayBuffer>;
        attentionOptions: {
            seqLen: unknown;
            numHeads: unknown;
            numKVHeads: unknown;
            headDim: unknown;
            scale: number;
            causal: boolean;
        };
        projectionCaches: {
            q: {
                down: Float32Array<ArrayBuffer>;
                rank: unknown;
                scale: number;
            } | null;
            k: {
                down: Float32Array<ArrayBuffer>;
                rank: unknown;
                scale: number;
            } | null;
            v: {
                down: Float32Array<ArrayBuffer>;
                rank: unknown;
                scale: number;
            } | null;
            o: {
                down: Float32Array<ArrayBuffer>;
                rank: unknown;
                scale: number;
            } | null;
        };
    };
};
export function qwenFullAttentionModuleBackward(inputs: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
    hidden: Float32Array<ArrayBuffer>;
    lora: {
        q: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
        k: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
        v: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
        o: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
    };
};
declare function projectionBackward(input: unknown, weight: unknown, gradOutput: unknown, rows: unknown, inputSize: unknown, outputSize: unknown, adapter: unknown, cache: unknown): {
    input: Float32Array<ArrayBuffer>;
    A: null;
    B: null;
} | {
    input: Float32Array<ArrayBuffer>;
    A: Float32Array<ArrayBuffer>;
    B: Float32Array<ArrayBuffer>;
};
declare function projectionForward(input: unknown, weight: unknown, rows: unknown, inputSize: unknown, outputSize: unknown, adapter: unknown): {
    output: Float32Array<ArrayBuffer>;
    cache: null;
} | {
    output: Float32Array<ArrayBuffer>;
    cache: {
        down: Float32Array<ArrayBuffer>;
        rank: unknown;
        scale: number;
    };
};
declare function rmsNormOffsetBackward(input: unknown, weight: unknown, gradOutput: unknown, cache: unknown, rows: unknown, width: unknown): Float32Array<ArrayBufferLike>;
declare function rmsNormOffsetForward(input: unknown, weight: unknown, rows: unknown, width: unknown, eps: unknown): {
    output: Float32Array<ArrayBufferLike>;
    cache: {
        inverseRms: Float32Array<ArrayBufferLike>;
    };
};
export { projectionBackward as qwenFrozenLoraProjectionBackward, projectionForward as qwenFrozenLoraProjectionForward, rmsNormOffsetBackward as qwenRmsNormOffsetBackward, rmsNormOffsetForward as qwenRmsNormOffsetForward };
