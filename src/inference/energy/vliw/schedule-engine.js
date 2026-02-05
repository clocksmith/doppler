import { ENGINE_ORDER } from './constants.js';

export function computeEngineOffsets(caps) {
  const offsets = {};
  const slotEngines = [];
  const slotIndices = [];
  let cursor = 0;
  ENGINE_ORDER.forEach((engine) => {
    offsets[engine] = cursor;
    const cap = Math.max(0, caps[engine] || 0);
    for (let i = 0; i < cap; i++) {
      slotEngines.push(engine);
      slotIndices.push(i);
    }
    cursor += cap;
  });
  return {
    offsets,
    totalSlots: cursor,
    slotEngines,
    slotIndices,
  };
}
