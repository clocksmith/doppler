export declare function requireObject(value: unknown, label: string, errors: string[]): boolean;
export declare function requireExactKeys(value: unknown, allowed: ReadonlySet<string>, label: string, errors: string[]): void;
export declare function requireString(value: unknown, label: string, errors: string[]): void;
export declare function requireDigest(value: string | null | undefined, label: string, errors: string[]): void;
