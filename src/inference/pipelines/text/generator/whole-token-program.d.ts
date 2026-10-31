export declare const WHOLE_TOKEN_CAPTURE_STEPS: readonly number[];

export type WholeTokenWordModel =
  | { kind: 'const'; value: number }
  | { kind: 'token' }
  | { kind: 'affine'; base: number; slope: number; pos0: number };

/** Classify one u32 word observed in three consecutive decode steps. */
export declare function classifyWord(
  values: readonly [number, number, number] | number[],
  positions: readonly number[],
  tokens: readonly number[]
): WholeTokenWordModel | null;

export declare class WholeTokenProgram {
  constructor(device: GPUDevice);
  readonly ready: boolean;
  readonly dispatchCount: number;
  readonly stats: { observedSteps: number; replayedTokens: number };
  observe<T>(pos: number, token: number, run: (attach: (recorder: import('../../../../gpu/command-recorder.js').CommandRecorder) => void) => Promise<T>): Promise<T>;
  run(pos: number, token: number): Promise<Uint32Array>;
  destroy(): void;
}
