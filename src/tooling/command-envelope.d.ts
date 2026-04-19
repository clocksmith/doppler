import type { ToolingCommandRequest } from './command-api.js';

export declare const TOOLING_ENVELOPE_SCHEMA_VERSION: number;

export interface ToolingCommandErrorDetails {
  command?: string | null;
  workload?: string | null;
  suite?: string | null;
  workloadType?: string | null;
  modelId?: string | null;
  requestedWorkload?: string | null;
  allowedWorkloads?: string[] | null;
  requestedSuite?: string | null;
  allowedSuites?: string[] | null;
  fromSurface?: string | null;
  toSurface?: string | null;
  surface?: string | null;
  [key: string]: unknown;
}

export interface ToolingSuccessEnvelope<T = unknown> {
  ok: true;
  schemaVersion: number;
  surface: string;
  request: ToolingCommandRequest;
  result: T;
  meta?: Record<string, unknown> | null;
}

export interface ToolingErrorPayload {
  code: string;
  message: string;
  details: ToolingCommandErrorDetails | null;
  retryable: boolean | null;
}

export interface ToolingErrorEnvelope {
  ok: false;
  schemaVersion: number;
  surface: string | null;
  request: ToolingCommandRequest | null;
  error: ToolingErrorPayload;
}

export interface ToolingCommandErrorOptions {
  code?: string | null;
  details?: ToolingCommandErrorDetails | null;
  retryable?: boolean | null;
  cause?: unknown;
}

export declare class ToolingCommandError extends Error {
  code: string;
  details: ToolingCommandErrorDetails | null;
  retryable: boolean | null;
  constructor(message: string, options?: ToolingCommandErrorOptions);
}

export declare function normalizeToToolingCommandError(
  error: unknown,
  context?: {
    surface?: string | null;
    request?: ToolingCommandRequest | null;
  }
): ToolingCommandError;

export declare function createToolingSuccessEnvelope<T = unknown>(input: {
  surface: string;
  request: ToolingCommandRequest;
  result: T;
  meta?: Record<string, unknown> | null;
}): ToolingSuccessEnvelope<T>;

export declare function createToolingErrorEnvelope(
  error: unknown,
  context?: {
    surface?: string | null;
    request?: ToolingCommandRequest | null;
  }
): ToolingErrorEnvelope;

export declare function isToolingSuccessEnvelope(value: unknown): value is ToolingSuccessEnvelope;
export declare function isToolingErrorEnvelope(value: unknown): value is ToolingErrorEnvelope;
