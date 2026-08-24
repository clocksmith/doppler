export declare function createAbortError(message?: string): Error;
export declare function fetchWithRetry(
  url: string | URL,
  options?: RequestInit
): Promise<Response>;

