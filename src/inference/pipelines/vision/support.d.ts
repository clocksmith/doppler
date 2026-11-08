/**
 * Which declared vision architectures the runtime can execute faithfully.
 *
 * @module inference/pipelines/vision/support
 */

export interface VisionArchitectureSupportEntry {
  executable: boolean;
  reason: string | null;
}

export declare const VISION_ARCHITECTURE_SUPPORT: Readonly<{
  gemma4: Readonly<VisionArchitectureSupportEntry>;
  glmocr: Readonly<VisionArchitectureSupportEntry>;
  qwen3vl: Readonly<VisionArchitectureSupportEntry>;
}>;

export declare function resolveVisionArchitectureSupport(visionArchitecture: unknown): {
  architecture: string;
  declared: boolean;
  executable: boolean;
  reason: string | null;
};

export declare function executableVisionArchitectures(): string[];

export declare function describeVisionUnavailable(
  state: { visionUnsupportedReason?: string | null } | null | undefined,
  operation: string
): string;

export declare function resolveVisionCapability(
  manifest: { image_token_id?: unknown; quantizationInfo?: { vision?: unknown } | null } | null | undefined,
  modelConfig: { visionConfig?: { visionArchitecture?: unknown } | null } | null | undefined
): {
  state: {
    visionCapable: boolean;
    imageTokenId: number | null;
    visionConfig: object | null;
    visionUnsupportedReason: string | null;
  };
  info: string | null;
  warning: string | null;
};
