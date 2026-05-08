import type { CatalogModelLane } from '../../config/model-lanes/catalog-lane-resolver.js';

export interface KernelCapabilities {
  hasF16?: boolean;
  hasSubgroups?: boolean;
  hasSubgroupsF16?: boolean;
  maxBufferSize?: number;
  maxWorkgroupSize?: number;
  maxWorkgroupStorageSize?: number;
  adapterInfo?: {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
  };
  [key: string]: unknown;
}

export interface ExecutionLaneSelection {
  entry: CatalogModelLane;
  selectedModelId: string | null;
  usedFallback: boolean;
  rejected: Array<{
    modelId: string | null;
    reason: string;
  }>;
  reason: string;
}

export declare function hasF16SubgroupLaneSupport(capabilities: KernelCapabilities | null | undefined): boolean;

export declare function assertExecutionLaneManifestSupported(
  entry: CatalogModelLane | null | undefined,
  manifest: Record<string, unknown>,
  capabilities: KernelCapabilities | null | undefined,
  options?: {
    normalizeManifest?: (manifest: Record<string, unknown>) => Record<string, unknown>;
    kernelPathPolicy?: unknown;
    platform?: {
      id?: string;
      vendor?: string;
      architecture?: string;
    };
  }
): void;

export declare function selectExecutionLaneForCapabilities(
  entry: CatalogModelLane,
  options?: {
    capabilities?: KernelCapabilities | null;
    manifestByModelId?: Map<string, Record<string, unknown>> | Record<string, Record<string, unknown>>;
    manifests?: Map<string, Record<string, unknown>> | Record<string, Record<string, unknown>>;
    normalizeManifest?: (manifest: Record<string, unknown>) => Record<string, unknown>;
    kernelPathPolicy?: unknown;
    platform?: {
      id?: string;
      vendor?: string;
      architecture?: string;
    };
  }
): ExecutionLaneSelection;
