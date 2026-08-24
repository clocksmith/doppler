// AbortSignal contract: public inference primitives reject with AbortError at
// checkpoints between dispatches. WebGPU does not expose mid-dispatch cancel.
export class AbortError extends Error {
  constructor(message = 'Doppler: aborted') {
    super(message);
    this.name = 'AbortError';
    this.code = 'ABORT_ERR';
  }
}

export function isAbortError(error) {
  return !!error
    && (error.name === 'AbortError' || error.code === 'ABORT_ERR' || error.code === 20);
}

export function assertNotAborted(signal) {
  if (signal?.aborted) {
    throw new AbortError(
      typeof signal.reason === 'string' ? signal.reason : 'Doppler: aborted'
    );
  }
}
