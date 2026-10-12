export declare function recurrentStateComponents(precision: unknown): 1 | 2;
export declare function resolveRecurrentStateLayout(sessionSettings?: { linearRecurrentStatePrecision?: 'f32' | 'two-f32' | null } | null): { precision: 'f32' | 'two-f32'; components: 1 | 2 };
export declare function assertRecurrentStateLayout(state: {
  recurrentStatePrecision: unknown;
  recurrentStateComponents: number;
  numVHeads: number;
  headKDim: number;
  headVDim: number;
  recurrentState: Float32Array;
}): 1 | 2;
