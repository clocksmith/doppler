import type { DopplerRuntimeSession } from '../../client/runtime/composition-root.js';
import type { PackServePolicy } from '../../config/pack-serve.js';
export type { PackServePolicy } from '../../config/pack-serve.js';
/** Structural Node HTTP boundary; consumers of other exports need no Node typings. */
export interface PackHttpEvents {
  once(event: string, listener: (...args: unknown[]) => void): unknown;
  off(event: string, listener: (...args: unknown[]) => void): unknown;
}
export interface PackHttpRequest extends PackHttpEvents {
  method?: string;
  url?: string;
  headers: Record<string, string | string[] | undefined>;
  on(event: string, listener: (...args: unknown[]) => void): unknown;
  pause(): unknown;
}
export interface PackHttpResponse extends PackHttpEvents {
  headersSent: boolean;
  destroyed: boolean;
  writableEnded: boolean;
  setHeader(name: string, value: string): unknown;
  writeHead(statusCode: number, headers: Record<string, string>): unknown;
  write(chunk: string): boolean;
  end(chunk?: string): unknown;
}
export interface PackServeHandler {
  (req: PackHttpRequest, res: PackHttpResponse): Promise<void>;
  close(): Promise<void>;
}
export function createPackServeHandler(options: {
  session: DopplerRuntimeSession;
  policy: PackServePolicy;
  token: string;
}): PackServeHandler;
