// ============================================================================
// Kernel Rule Registry
//
// This module intentionally duplicates the rule-registry pattern from
// src/rules/rule-registry.js. The separation is deliberate: kernel call
// sites use this thin wrapper so that every kernel-domain rule lookup
// automatically enriches the context with GPU capabilities (hasF16,
// hasSubgroups, etc.) and platform detection metadata. Merging back into
// the base rule-registry would force non-kernel callers to depend on GPU
// device state, breaking the layered-config contract.
//
// Canonical base rule registry: src/rules/rule-registry.js
// ============================================================================
import { getRuleSet as getBaseRuleSet, selectRuleValue as selectBaseRuleValue } from '../../rules/rule-registry.js';
import { getDeviceEpoch, getKernelCapabilities, getPlatformConfig } from '../device.js';

let cachedKernelRuleContextBase = null;
let cachedKernelRuleContextEpoch = null;

export function getRuleSet(group, name) {
  return getBaseRuleSet('kernels', group, name);
}

function getKernelRuleContextBase() {
  let epoch = null;
  try {
    epoch = getDeviceEpoch();
  } catch {
    epoch = null;
  }

  if (
    epoch !== null
    && cachedKernelRuleContextEpoch === epoch
    && cachedKernelRuleContextBase
  ) {
    return cachedKernelRuleContextBase;
  }

  const base = {};
  let cacheable = false;

  let capabilities = null;
  try {
    capabilities = getKernelCapabilities();
  } catch {
    capabilities = null;
  }
  if (capabilities) {
    base.hasSubgroups = capabilities.hasSubgroups;
    base.hasSubgroupsF16 = capabilities.hasSubgroupsF16;
    base.hasF16 = capabilities.hasF16;
    cacheable = true;
  }

  let platform = null;
  try {
    platform = getPlatformConfig()?.platform ?? null;
  } catch {
    platform = null;
  }
  if (platform) {
    base.platformId = platform.id;
    base.platformVendor = platform.detection?.vendor ?? null;
    base.platformArchitecture = platform.detection?.architecture ?? null;
    base.platformDevice = platform.detection?.device ?? null;
  }

  if (epoch !== null && cacheable) {
    cachedKernelRuleContextEpoch = epoch;
    cachedKernelRuleContextBase = Object.freeze(base);
    return cachedKernelRuleContextBase;
  }

  return base;
}

function enrichKernelRuleContext(context) {
  const base = getKernelRuleContextBase();
  const next = { ...base };
  if (!context || typeof context !== 'object') {
    return next;
  }
  for (const [key, value] of Object.entries(context)) {
    if (value !== undefined || !(key in base)) {
      next[key] = value;
    }
  }
  return next;
}

export function selectRuleValue(group, name, context) {
  return selectBaseRuleValue('kernels', group, name, enrichKernelRuleContext(context));
}
