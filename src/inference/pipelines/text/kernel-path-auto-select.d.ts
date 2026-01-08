import type { KernelPathRef } from '../../../config/schema/index.js';
import type { KernelPathSource } from '../../../config/kernel-path-loader.js';
import type { KernelCapabilities } from '../../../gpu/device.js';

export function resolveCapabilityKernelPathRef(
  configuredKernelPathRef: KernelPathRef,
  kernelPathSource: KernelPathSource,
  capabilities?: Partial<KernelCapabilities> | null
): KernelPathRef;
