export function normalizeTensorSourceTransform(location: any, tensorName: any, options?: {}): {
    kind: string;
    scheme: any;
    sourceDtype: string;
    targetDtype: string;
    scale: number;
    zeroPoint: number;
} | {
    rowSumSource?: any;
    scaleSource: any;
    scaleDivisor?: number | undefined;
    scaleSemantics: string;
    kind: string;
    scheme: any;
    sourceDtype: string;
    targetDtype: string;
    storageEncoding: string;
} | {
    sumSource?: any;
    scaleCompanionDtype?: undefined;
    scaleCompanionDequant?: undefined;
    storageShape: number[];
    quantAxis: number;
    scaleSource: any;
    scaleDivisor?: number | undefined;
    scaleSemantics: string;
    kind: string;
    scheme: any;
    sourceDtype: string;
    targetDtype: string;
    storageEncoding: string;
} | {
    sumSource?: any;
    scaleCompanionDtype: string;
    scaleCompanionDequant: {
        scale: number;
        zeroPoint: number;
    };
    storageShape: number[];
    quantAxis: number;
    scaleSource: any;
    scaleDivisor?: number | undefined;
    scaleSemantics: string;
    kind: string;
    scheme: any;
    sourceDtype: string;
    targetDtype: string;
    storageEncoding: string;
} | {
    sumSource?: any;
    storageShape: number[];
    quantAxis: number;
    storageBlockSize: number;
    storageLaneOrder: any;
    scaleSource: any;
    scaleDivisor?: number | undefined;
    scaleSemantics: string;
    kind: string;
    scheme: any;
    sourceDtype: string;
    targetDtype: string;
    storageEncoding: string;
} | null | undefined;
