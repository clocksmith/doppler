export interface ProductionReleaseCommandRequest {
  action: 'qualify' | 'decide';
  manifestPath: string;
  outputDirectory: string;
  repoRoot?: string | null;
  forgeConfigPath?: string | null;
  targetId?: string | null;
  deviceIdentityPath?: string | null;
  fleetReceiptPaths?: string[];
  packTrustedSignersPath: string;
  fleetTrustedSignersPath?: string | null;
  signingPrivateKeyPath: string;
  signingPublicKeyPath: string;
  signingAuthority: string;
}

export declare function runProductionRelease(
  request: ProductionReleaseCommandRequest
): Promise<Record<string, unknown>>;
