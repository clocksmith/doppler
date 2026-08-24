export function preprocessGemma4Image(pixels: any, width: any, height: any, visionConfig: any, softTokenBudget: any): {
    patches: Float32Array<ArrayBuffer>;
    positions: Int32Array<ArrayBuffer>;
    gridHeight: number;
    gridWidth: number;
    numPatches: number;
    outputLength: number;
};
export function encodeGemma4Image(params: any): Promise<{
    features: GPUBuffer;
    numTokens: number;
    gridThw: number[];
    imageWidth: number;
    imageHeight: number;
}>;
