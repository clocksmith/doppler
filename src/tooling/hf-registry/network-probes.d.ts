export declare function probeUrl(
  url: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function fetchJson(
  url: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function fetchRepoHeadSha(
  repoId: string,
  options?: Record<string, unknown>
): Promise<string | null>;
