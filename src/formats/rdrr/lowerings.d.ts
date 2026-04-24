import type {
  IntegrityExtensionsLoweringEntry,
  RDRRManifest,
} from './types.js';

export declare const DOPPLER_LOWERING_MISSING: 'DOPPLER_LOWERING_MISSING';
export declare const DOPPLER_LOWERING_REJECTED: 'DOPPLER_LOWERING_REJECTED';

export declare function findLowering(
  manifest: RDRRManifest,
  kernelRef: string,
  backend: string
): IntegrityExtensionsLoweringEntry | null;

export declare function isRejectionEntry(
  entry: IntegrityExtensionsLoweringEntry | null | undefined
): boolean;

export declare function findLoweringOrThrow(
  manifest: RDRRManifest,
  kernelRef: string,
  backend: string
): IntegrityExtensionsLoweringEntry;

export declare function listSupportedBackends(
  manifest: RDRRManifest,
  kernelRefs: string[]
): string[];
