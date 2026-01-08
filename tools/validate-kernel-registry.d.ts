export interface KernelRegistry {
  operations: Record<string, { variants: Record<string, { wgsl: string }> }>;
}

export declare function fileExists(filePath: string): Promise<boolean>;
export declare function main(): Promise<void>;
