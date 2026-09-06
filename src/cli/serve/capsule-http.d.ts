import type { CapsuleHttpRequest, CapsuleHttpResponse } from './capsule-handler.js';
import type { CapsuleServePolicy } from '../../config/capsule-serve.js';
export class CapsuleServeError extends Error {
  code: string;
  statusCode: number;
  constructor(code: string, message: string, statusCode: number);
}
export function createCapsuleRequestAuthorization(token: string, policy: Readonly<CapsuleServePolicy>): (req: CapsuleHttpRequest, res: CapsuleHttpResponse) => void;
export function readCapsuleRequest(req: CapsuleHttpRequest, maxBytes: number, signal: AbortSignal): Promise<unknown>;
export function writeCapsuleEvent(res: CapsuleHttpResponse, event: string, signal: AbortSignal): Promise<void>;
export function endCapsuleError(res: CapsuleHttpResponse, error: unknown, remainingBytes: number): void;
