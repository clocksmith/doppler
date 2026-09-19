export function qwenLinearDecoderLayerForward(inputs: unknown, options: unknown): {
    output: Float32Array<ArrayBuffer>;
    finalState: unknown;
    cache: {
        inputNorm: {
            output: Float32Array<ArrayBufferLike>;
            cache: {
                inverseRms: Float32Array<ArrayBufferLike>;
            };
        };
        attentionInputs: unknown;
        attention: unknown;
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
export function qwenLinearDecoderLayerBackward(inputs: unknown, gradOutput: unknown, cache: unknown, options: unknown): {
    hidden: Float32Array<ArrayBuffer>;
    initialState: unknown;
    lora: {
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
