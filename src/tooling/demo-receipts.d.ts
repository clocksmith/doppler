export declare const DEMO_CONTRACT_RECEIPT_SCHEMA: 'doppler.demo-contract-receipt/v1';
export declare const DEMO_HARDWARE_RECEIPT_SCHEMA: 'doppler.demo-hardware-receipt/v1';

export interface DopplerDemoContractReceipt {
  schema: 'doppler.demo-contract-receipt/v1';
  status: 'passed' | 'failed';
  createdAtUtc: string;
  entrypoint: '/demo/index.html';
  executionClass: 'mocked-contract' | 'software-webgpu' | 'hardware-webgpu';
  journey: {
    catalogRendered: boolean;
    modelSelected: boolean;
    modelLoaded: boolean;
    generationCompleted: boolean;
  };
  shellManifestDigest: string;
  fatalConsoleErrors: string[];
}

export interface DopplerDemoHardwareReceipt {
  schema: 'doppler.demo-hardware-receipt/v1';
  status: 'passed' | 'failed' | 'capability-skip';
  createdAtUtc: string;
  executionClass: 'software-webgpu' | 'hardware-webgpu';
  browser: Record<string, unknown>;
  adapter: Record<string, unknown>;
  artifact: {
    modelId: string;
    manifestHash: string;
  };
  online: {
    outputText: string;
    tokenIds: number[];
    transcriptHash: string;
  };
  offline: {
    outputText: string;
    tokenIds: number[];
    transcriptHash: string;
  };
  lifecycle: {
    allPagesClosed: boolean;
    networkDisabled: boolean;
    persistentCacheRestored: boolean;
    upgradeChecked: boolean;
    partialCacheFailureChecked: boolean;
  };
  fingerprint: Record<string, unknown>;
}

export declare function validateDemoContractReceipt(
  receipt: DopplerDemoContractReceipt
): DopplerDemoContractReceipt;
export declare function validateDemoHardwareReceipt(
  receipt: DopplerDemoHardwareReceipt
): DopplerDemoHardwareReceipt;
