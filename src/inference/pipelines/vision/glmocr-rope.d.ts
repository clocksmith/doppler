export interface GlmOcrRopePositionPlan {
  temporal: Int32Array;
  height: Int32Array;
  width: Int32Array;
  promptLength: number;
  capacity: number;
  ropeDelta: number;
}

export declare function buildGlmOcrRopePositionPlan(options: {
  promptLength: number;
  capacity: number;
  imageStartOffset: number;
  imageTokenLength: number;
  gridHeight: number;
  gridWidth: number;
  mergeSize: number;
}): GlmOcrRopePositionPlan;

export declare function uploadGlmOcrRopeFrequencies(
  positionPlan: GlmOcrRopePositionPlan,
  options: {
    rotaryDim: number;
    frequencyBaseDim: number;
    ropeTheta: number;
    mropeSection: [number, number, number];
  }
): Promise<{
  cos: GPUBuffer;
  sin: GPUBuffer;
  positionPlan: GlmOcrRopePositionPlan;
  release(): void;
}>;
