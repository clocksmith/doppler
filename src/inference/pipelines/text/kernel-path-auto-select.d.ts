import type { KernelPathRef } from '../../../config/schema/index.js';
import type { KernelPathPolicy, KernelPathSource } from '../../../config/kernel-path-loader.js';
import type { KernelCapabilities } from '../../../gpu/device.js';

export function resolveKernelPathPolicy(policy?: Partial<KernelPathPolicy> | null): KernelPathPolicy;

export function resolveCapabilityKernelPathRef(
  configuredKernelPathRef: KernelPathRef,
  kernelPathSource: KernelPathSource,
  capabilities?: Partial<KernelCapabilities> | null,
  kernelPathPolicy?: Partial<KernelPathPolicy> | null
): KernelPathRef;
