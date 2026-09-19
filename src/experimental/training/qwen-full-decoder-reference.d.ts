export function qwenFullDecoderLayerForward(inputs: unknown, options: unknown): {
    output: Float32Array<ArrayBuffer>;
    cache: {
        inputNorm: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                inverseRms: Float32Array<ArrayBufferLike>;
            };
        };
        attentionInputs: unknown;
        attention: {
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
        postAttention: Float32Array<ArrayBuffer>;
        normalizedPostAttention: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                inverseRms: Float32Array<ArrayBufferLike>;
            };
        };
        gate: {
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
        up: {
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
        activated: Float32Array<ArrayBuffer>;
        down: {
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
    };
};
export function qwenFullDecoderLayerBackward(inputs: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
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
        gate: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
        up: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
        down: {
            A: Float32Array<ArrayBuffer> | null;
            B: Float32Array<ArrayBuffer> | null;
        };
    };
};
