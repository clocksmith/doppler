export interface WholeTokenBatchStream {
  readonly rows: AsyncIterable<Uint32Array>;
  /** Wait for every queued step and release all readback/feedback buffers. Always call in finally. */
  settle(): Promise<void>;
}

/** Consecutive consistent decode steps observed before a program is built. */
export declare const WHOLE_TOKEN_OBSERVED_STEPS: number;

/** Explicit GPU-fed window. Word zero of its readback is the emitted token count. */
export interface GPUWindowInput { readonly tokenInput: GPUBuffer; readonly maxOutputTokens: number }
export interface GPUWindowExecution {
  /** Same execution plan and resources as the observed windows. */
  readonly identity: readonly unknown[];
  /** Record a rollback checkpoint before any window commands, using the same encoder. */
  recordPrefix(encoder: GPUCommandEncoder): void;
  /** Called immediately after successful submission, including when a later map fails. */
  onSubmitted(): void;
}

export declare class WholeTokenProgram {
  constructor(device: GPUDevice, gpuWindow?: GPUWindowInput | null);
  readonly ready: boolean;
  /** A kept program whose next observed step must match it (verify) before it replays. */
  readonly awaitingVerification: boolean;
  readonly dispatchCount: number;
  /** Owned replay resources only, excluding borrowed model weights and caches. */
  readonly privateBytes: number;
  readonly positionRange: { owner: string; min: number; max: number };
  readonly stats: {
    observedSteps: number;
    replayedTokens: number;
    /** Submitted replay command buffers; ordinary observations have their own recorder counts. */
    replaySubmissions: number;
    batchReadbacks: number;
    restarts: number;
    invalidations: number;
    /** This generation replays a program kept from an earlier one. */
    reused: boolean;
    /** The kept program was verified before step 1 ran (verifyAhead), and step 1 replays it. */
    verifiedAhead: boolean;
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
  /**
   * Verify the kept program on an ordinary step recorded at `pos` but never submitted: `run`
   * stops before the submit by throwing `stopSignal`. True when the program matches and
   * replays from `pos`; false when it was invalidated (the observation is discarded).
   */
  verifyAhead(
    pos: number,
    token: number,
    run: (attach: (recorder: import('../../../../gpu/command-recorder.js').CommandRecorder | null) => void) => Promise<unknown>,
    identityOf: () => readonly unknown[],
    limits: readonly import('./whole-token-observation.js').PositionLimit[],
    stopSignal: unknown
  ): Promise<boolean>;
  /** Why the built program cannot run the step at `pos` with these resources, or null. */
  rejects(pos: number, identity: readonly unknown[]): string | null;
  /** Drop the built program so the following steps are observed again. */
  invalidate(reason: string): void;
  run(pos: number, token: number): Promise<Uint32Array>;
  /** Returns untrusted compact feedback. Owner must validate it before publication and recover on failure. */
  runWindow(pos: number, execution: GPUWindowExecution): Promise<Uint32Array>;
  /** Supply the final observation's validated count before the first replay. */
  acceptObservedWindow(pos: number, count: number): void;
  startBatch(pos: number, token: number, count: number): Promise<WholeTokenBatchStream>;
  /** Exact sequential GPU steps; caller must retain a state checkpoint through publication. */
  runBatch(pos: number, token: number, count: number): Promise<Uint32Array[]>;
  /** Called only after restoring the matching KV/recurrent checkpoint. */
  resumeAfterRestore(pos: number): void;
  destroy(): void;
}
