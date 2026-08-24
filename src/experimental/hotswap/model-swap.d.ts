export function swapModel(currentModel: any, newModelLoader: any, policy?: {}, context?: {}): Promise<{
    swapped: boolean;
    model: any;
    decision: import("./runtime.js").HotSwapRolloutDecision;
}>;
