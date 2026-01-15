export declare function parseJsonl(text: string): unknown[];

export declare function loadJsonl(
  source: string,
  options?: { fetch?: (url: string) => Promise<string> }
): Promise<unknown[]>;

export declare function mapJsonl<T, U>(
  records: T[],
  mapper?: (record: T) => U | null | undefined
): U[];
