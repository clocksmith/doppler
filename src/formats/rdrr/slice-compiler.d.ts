export interface CompiledTensorByteRange {
  shardIndex: number;
  byteStart: number;
  byteEnd: number;
}

export interface CompiledCompanionRange {
  role: string;
  tensorId: string;
  byteRanges: CompiledTensorByteRange[];
}

export interface CompiledTensorSlice {
  tensorId: string;
  axis: number;
  rangeStart: number;
  rangeEnd: number;
  byteRanges: CompiledTensorByteRange[];
  companionRanges: CompiledCompanionRange[];
}

export declare function compileTensorSlice(options: {
  tensorMap: Record<string, unknown>;
  tensorId: string;
  axis: number;
  rangeStart: number;
  rangeEnd: number;
}): CompiledTensorSlice;

