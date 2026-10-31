/** Consecutive consistent decode steps observed before a program is built. */
export declare const WHOLE_TOKEN_OBSERVED_STEPS: number;

export declare class WholeTokenProgram {
  constructor(device: GPUDevice);
  readonly ready: boolean;
  /** A kept program whose next observed step must match it (verify) before it replays. */
  readonly awaitingVerification: boolean;
  readonly dispatchCount: number;
  readonly positionRange: { owner: string; min: number; max: number };
  readonly stats: {
    observedSteps: number;
    replayedTokens: number;
    restarts: number;
    invalidations: number;
    /** This generation replays a program kept from an earlier one. */
    reused: boolean;
    lastInvalidation: string | null;
  };
  observe<T>(
    pos: number,
    token: number,
    run: (attach: (recorder: import('../../../../gpu/command-recorder.js').CommandRecorder | null) => void) => Promise<T>,
    identityOf: () => readonly unknown[],
    limits?: readonly import('./whole-token-observation.js').PositionLimit[]
  ): Promise<T>;
  /** Keep the built program for the next generation; its first observed step must verify it. */
  beginGeneration(): void;
  /**
   * Observe the kept program's first step of a generation and accept the program only if
   * that step builds the same template with the steps it was built from; otherwise
   * invalidate it and start observing from this step.
   */
  verify<T>(
    pos: number,
    token: number,
    run: (attach: (recorder: import('../../../../gpu/command-recorder.js').CommandRecorder | null) => void) => Promise<T>,
    identityOf: () => readonly unknown[],
    limits?: readonly import('./whole-token-observation.js').PositionLimit[]
  ): Promise<T>;
  /** Why the built program cannot run the step at `pos` with these resources, or null. */
  rejects(pos: number, identity: readonly unknown[]): string | null;
  /** Drop the built program so the following steps are observed again. */
  invalidate(reason: string): void;
  run(pos: number, token: number): Promise<Uint32Array>;
  destroy(): void;
}
