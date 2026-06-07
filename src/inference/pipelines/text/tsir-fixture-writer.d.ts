export interface TsirFixture {
  dir: string;
  layerFilter?: number[] | null;
  prefillOnly?: boolean;
  pendingReads?: TsirPendingRead[];
  records?: TsirFixtureRecord[];
  qkvByLayer?: Map<number, unknown>;
}

export interface TsirPendingRead {
  stage: string;
  tsirStage: string;
  layerIdx: number;
  numTokens: number;
  hiddenSize: number;
  dtype: 'f16' | 'f32';
  carry: GPUBuffer;
  alignedBytes: number;
}

export interface TsirFixtureRecord {
  stage: string;
  tsirStage: string;
  layerIdx: number;
  filePath?: string;
  shape?: number[];
  dtype?: string;
  byteLength?: number;
  payloadByteLength?: number;
  written: boolean;
  note?: string;
}

export declare const TSIR_BOUNDARY_STAGES: string[];

export declare function mapStageToTsirBoundary(stage: string): string | null;

export declare function writeNpyF32(
  filePath: string,
  shape: readonly number[],
  data: Float32Array
): Promise<{ byteLength: number; payloadByteLength: number }>;

export declare function maybeWriteFixtureSnapshot(
  stage: string,
  buffer: Float32Array | GPUBuffer,
  options: {
    tsirFixture?: TsirFixture | null;
    layerIdx: number;
    numTokens: number;
    hiddenSize: number;
    dtype?: 'f16' | 'f32';
    recorder?: { getEncoder?(): GPUCommandEncoder } | null;
  }
): Promise<TsirFixtureRecord | null>;

export declare function drainPendingTsirReads(tsirFixture: TsirFixture | null | undefined): Promise<TsirFixtureRecord[]>;
