export interface ConversionRunTimingReceipt {
  startedAtUtc: string;
  completedAtUtc: string;
  durationMs: number;
}

export interface ConversionRunTiming {
  complete(): Readonly<ConversionRunTimingReceipt>;
}

export declare function createConversionRunTiming(options?: {
  now?: () => Date;
  monotonicNow?: () => number;
}): ConversionRunTiming;
