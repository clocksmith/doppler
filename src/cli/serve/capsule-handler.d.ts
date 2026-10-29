import type { DopplerRuntimeSession } from '../../client/runtime/composition-root.js';
import type { CapsuleServePolicy } from '../../config/capsule-serve.js';
export type { CapsuleServePolicy } from '../../config/capsule-serve.js';
/** Structural Node HTTP boundary; consumers of other exports need no Node typings. */
export interface CapsuleHttpEvents {
  once(event: string, listener: (...args: unknown[]) => void): unknown;
  off(event: string, listener: (...args: unknown[]) => void): unknown;
}
export interface CapsuleHttpRequest extends CapsuleHttpEvents {
  method?: string;
  url?: string;
  headers: Record<string, string | string[] | undefined>;
  on(event: string, listener: (...args: unknown[]) => void): unknown;
  pause(): unknown;
}
export interface CapsuleHttpResponse extends CapsuleHttpEvents {
  headersSent: boolean;
  destroyed: boolean;
  writableEnded: boolean;
  setHeader(name: string, value: string): unknown;
  writeHead(statusCode: number, headers: Record<string, string>): unknown;
  write(chunk: string): boolean;
  end(chunk?: string): unknown;
}
export interface CapsuleServeHandler {
  (req: CapsuleHttpRequest, res: CapsuleHttpResponse): Promise<void>;
  close(): Promise<void>;
}
export function createCapsuleServeHandler(options: {
  session: DopplerRuntimeSession;
  policy: CapsuleServePolicy;
  token: string;
}): CapsuleServeHandler;
