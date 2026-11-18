export declare function readRecordingJson(file: Blob, options?: { signal?: AbortSignal; maxBytes?: number }): Promise<{ value: unknown; byteLength: number }>;
export declare function readRecordingBlob(response: Response, expectedBytes: number): Promise<Blob>;
