export declare function computeElementCount(shape: readonly unknown[], tensorName: string): number;

export declare function computeSourceByteLength(
  elementCount: number,
  sourceDtype: string,
  tensorName: string
): number;

export declare function getPackedValuesPerByte(sourceDtype: string, tensorName: string): number;

export declare function validateLiteRTTransformTarget(
  location: { dtype?: unknown } | null | undefined,
  tensorName: string,
  transform: { targetDtype?: unknown },
  label: string
): void;

export declare function validateLiteRTStorageEncoding(
  storageEncoding: unknown,
  tensorName: string
): void;

export declare function validateLiteRTStorageLaneOrder(
  storageLaneOrder: unknown,
  storageBlockSize: number,
  tensorName: string
): void;

export declare function getLiteRTCompanionByteLength(
  companionSource: { size?: unknown } | null | undefined,
  tensorName: string,
  label: string,
  expectedByteLength: number
): number | null;
