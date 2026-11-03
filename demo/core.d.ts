export declare function onModelLoaded(model: unknown, modelId: string): void;
export declare function runGeneration(): Promise<void>;
export declare function stopGeneration(): void;
export declare function loadSampleInspection(): Promise<void>;

export declare function loadIllustrativeExample(): void;
/** Re-applies Branch threshold, fork switchers and availability to the token view. */
export declare function syncBranchView(): void;
/** Shows another branch of the latest answer. */
export declare function switchBranch(id: number): void;
/** Continues the latest answer from `tokenId` at token position `forkIndex`. */
export declare function runBranch(forkIndex: number, tokenId: number): Promise<void>;
