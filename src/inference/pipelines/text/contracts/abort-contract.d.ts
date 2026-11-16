import { DopplerErrorBase } from '../../../../errors/doppler-error.js';
export declare class AbortError extends DopplerErrorBase {
  code: 'ABORT_ERR';
  constructor(message?: string);
}

export declare function isAbortError(error: unknown): boolean;
export declare function assertNotAborted(signal?: AbortSignal | null): void;
