export declare function readStoredQuantizedValue(
  bytes: Uint8Array,
  index: number,
  sourceDtype: string,
  storageEncoding?: string
): number;

export declare function readQuantizedValue(
  bytes: Uint8Array,
  index: number,
  sourceDtype: string,
  storageEncoding?: string
): number;

export declare function computeStoredQuantizedSum(
  bytes: Uint8Array,
  sourceDtype: string,
  storageEncoding?: string
): number;

export declare function resolveLiteRTScaleValue(
  storedScale: number,
  transform: { scaleSemantics?: unknown; scaleDivisor?: unknown },
  tensorName: string,
  rowLabel: string
): number;
