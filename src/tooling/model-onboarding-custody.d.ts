export interface OnboardingFileRecord { path: string; digest: string; sizeBytes: number }
export declare function hashOnboardingFile(filename: string): Promise<string>;
export declare function retainOnboardingJson(outputDir: string, filename: string, value: unknown): Promise<{ path: string; digest: string }>;
export declare function inventoryOnboardingFiles(directory: string, root?: string): Promise<OnboardingFileRecord[]>;
