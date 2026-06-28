export declare function resolveBrowserModelUrl(
  request: Record<string, unknown>,
  browserOptions?: Record<string, unknown>
): Promise<Record<string, unknown>>;

export declare function resolveNodeModelUrl(
  request: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;

export declare function buildRequest(
  parsed: Record<string, unknown>,
  policy?: Record<string, unknown>
): Promise<{
  request: Record<string, unknown>;
  runConfig: Record<string, unknown>;
  surface: string;
}>;

export declare function createCliToolingErrorEnvelope(
  error: unknown,
  context?: Record<string, unknown>
): Record<string, unknown>;

export declare function finalizeCliCommandResponse(
  response: Record<string, unknown>,
  request: Record<string, unknown>
): Record<string, unknown>;

export declare function withJsonStdoutIsolation<T>(
  enabled: boolean,
  callback: () => T | Promise<T>
): Promise<T>;

export declare function performIntake(options?: {
  convertConfigValue?: string | Record<string, unknown> | null;
  manifestFlag?: string | null;
  modelDir?: string | null;
  skipConvert?: boolean;
}): Promise<{
  report: Record<string, unknown>;
  manifestPath: string | null;
  manifest: Record<string, unknown> | null;
}>;

export declare function checkCapturePrecondition(surface: string): Promise<Record<string, unknown>>;
