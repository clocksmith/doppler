import { isPlainObject } from '../../formats/plain-object.js';

/** @type {import('./parameter-groups.js').resolveModelParamGroups} */
export function resolveModelParamGroups(model) {
  if (model && typeof model.paramGroups === 'function') {
    const groups = model.paramGroups();
    if (!groups || typeof groups !== 'object') {
      throw new Error('model.paramGroups() must return an object of tensor arrays.');
    }
    return groups;
  }
  if (model && typeof model.loraParams === 'function') {
    return { lora: model.loraParams() };
  }
  return {};
}

/** @type {import('./parameter-groups.js').selectTrainableParamGroups} */
export function selectTrainableParamGroups(paramGroups, freezeMap) {
  /** @type {Record<string, unknown[]>} */
  const trainableGroups = {};
  const frozenGroups = [];
  for (const [groupName, params] of Object.entries(paramGroups)) {
    const normalizedParams = Array.isArray(params) ? params.filter(Boolean) : [];
    if (freezeMap?.[groupName] === true) {
      frozenGroups.push(groupName);
      continue;
    }
    trainableGroups[groupName] = normalizedParams;
  }
  return { trainableGroups, frozenGroups };
}

/** @type {import('./parameter-groups.js').isTensorLike} */
export function isTensorLike(value) {
  return isPlainObject(value)
    && Array.isArray(value.shape)
    && value.buffer != null;
}
