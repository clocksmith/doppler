export interface LeanExecutionContractRunResult {
  ok: boolean;
  toolchainRef: string | null;
  generatedPath: string;
  moduleName: string;
  facts: Record<string, unknown>;
}

export declare function resolveLeanBinary(): string;

export declare function runLeanCheck(options: {
  sourcePath: string;
  rootDir: string;
}): {
  ok: boolean;
  toolchainRef: string;
};

export declare function writeExecutionContractLeanModuleForManifest(
  manifest: Record<string, unknown>,
  options?: {
    rootDir?: string;
    moduleName?: string;
    emitPath?: string | null;
  }
): {
  rootDir: string;
  facts: Record<string, unknown>;
  moduleName: string;
  source: string;
  generatedPath: string;
  tempDir: string | null;
};

export declare function runLeanExecutionContractForManifest(
  manifest: Record<string, unknown>,
  options?: {
    rootDir?: string;
    moduleName?: string;
    emitPath?: string | null;
    check?: boolean;
  }
): LeanExecutionContractRunResult;
