export interface KernelValidatorFs {
  readText(path: string): Promise<string>;
  exists(path: string): Promise<boolean>;
  list(path: string): Promise<string[]>;
}

export interface KernelRegistryValidationResult {
  ok: boolean;
  errors: string[];
  filesCount: number;
}

export interface KernelOverrideLintResult {
  ok: boolean;
  errors: string[];
  checkedCount: number;
}

export declare class KernelValidator {
  constructor(fs: KernelValidatorFs);

  validateRegistry(registryPath: string, kernelDir: string): Promise<KernelRegistryValidationResult>;
  lintWgslOverrides(kernelDir: string, whitelist?: string[]): Promise<KernelOverrideLintResult>;
}
