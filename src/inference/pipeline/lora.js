

export { LORA_MODULE_ALIASES } from './lora-types.js';


export const getLoRAModule = (adapter, layerIdx, moduleName) => {
  if (!adapter) return null;
  if (adapter.targetModules && !adapter.targetModules.includes(moduleName)) return null;
  const layer = adapter.layers.get(layerIdx);
  if (!layer) return null;
  return layer[moduleName] || null;
};
