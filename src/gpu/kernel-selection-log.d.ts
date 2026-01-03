export declare function logKernelSelectionOnce(
  operation: string,
  payload: { variant?: string; reason?: string }
): void;

export declare function resetKernelSelectionLog(): void;

export declare function getKernelSelectionLog(): Array<{
  operation: string;
  variant: string;
  reason: string | null;
}>;
