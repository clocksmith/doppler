import type { PackHttpRequest, PackHttpResponse } from './pack-handler.js';
import type { PackServePolicy } from '../../config/pack-serve.js';
export class PackServeError extends Error {
  code: string;
  statusCode: number;
  constructor(code: string, message: string, statusCode: number);
}
export function createPackRequestAuthorization(token: string, policy: Readonly<PackServePolicy>): (req: PackHttpRequest, res: PackHttpResponse) => void;
export function readPackRequest(req: PackHttpRequest, maxBytes: number, signal: AbortSignal): Promise<unknown>;
export function writePackEvent(res: PackHttpResponse, event: string, signal: AbortSignal): Promise<void>;
export function endPackError(res: PackHttpResponse, error: unknown, remainingBytes: number): void;
