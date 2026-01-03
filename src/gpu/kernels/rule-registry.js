import { getRuleSet as getBaseRuleSet, selectRuleValue as selectBaseRuleValue } from '../../rules/rule-registry.js';
import { getKernelCapabilities, getPlatformConfig } from '../device.js';

export function getRuleSet(group, name) {
  return getBaseRuleSet('kernels', group, name);
}

function enrichKernelRuleContext(context) {
  const next = (context && typeof context === 'object') ? { ...context } : {};

  let capabilities = null;
  try {
    capabilities = getKernelCapabilities();
  } catch {
    capabilities = null;
  }
  if (capabilities) {
    if (next.hasSubgroups === undefined) next.hasSubgroups = capabilities.hasSubgroups;
    if (next.hasSubgroupsF16 === undefined) next.hasSubgroupsF16 = capabilities.hasSubgroupsF16;
    if (next.hasF16 === undefined) next.hasF16 = capabilities.hasF16;
  }

  let platform = null;
  try {
    platform = getPlatformConfig()?.platform ?? null;
  } catch {
    platform = null;
  }
  if (platform) {
    if (next.platformId === undefined) next.platformId = platform.id;
    if (next.platformVendor === undefined) next.platformVendor = platform.detection?.vendor ?? null;
    if (next.platformArchitecture === undefined) next.platformArchitecture = platform.detection?.architecture ?? null;
    if (next.platformDevice === undefined) next.platformDevice = platform.detection?.device ?? null;
  }

  return next;
}

export function selectRuleValue(group, name, context) {
  return selectBaseRuleValue('kernels', group, name, enrichKernelRuleContext(context));
}
