export declare function initXray(options?: { onChange?: () => void }): void;
export declare function updateXrayPanels(pipeline?: any): void;
export declare function resetXray(): void;
export declare function isXrayEnabled(): boolean;
export declare function isXrayProfilingNeeded(): boolean;
export declare function getXrayRuntimeNoticeText(options?: {
  tokenPressEnabled?: boolean;
  traceEnabled?: boolean;
  profilingEnabled?: boolean;
}): string | null;
