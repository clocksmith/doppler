

import { getManifest } from './parsing.js';
import { sortGroupIds, parseGroupExpertIndex } from './classification.js';

export function getGroup(groupId) {
  return getManifest()?.groups?.[groupId] ?? null;
}

export function getGroupIds() {
  return Object.keys(getManifest()?.groups || {});
}

export function getShardsForGroup(groupId) {
  return getManifest()?.groups?.[groupId]?.shards ?? [];
}

export function getTensorsForGroup(groupId) {
  return getManifest()?.groups?.[groupId]?.tensors ?? [];
}

export function getShardsForExpert(layerIdx, expertIdx) {
  const manifest = getManifest();
  const groupId = `layer.${layerIdx}.expert.${expertIdx}`;
  const group = manifest?.groups?.[groupId];
  if (group) {
    return group.shards;
  }
  throw new Error(`Missing expert group mapping: ${groupId}`);
}

export function getTensorsForExpert(layerIdx, expertIdx) {
  const manifest = getManifest();
  const groupId = `layer.${layerIdx}.expert.${expertIdx}`;
  const group = manifest?.groups?.[groupId];
  if (group) {
    return group.tensors;
  }
  throw new Error(`Missing expert group mapping: ${groupId}`);
}

export function getExpertBytes() {
  const manifest = getManifest();
  const expertGroups = Object.entries(manifest?.groups || {})
    .filter(([id]) => id.includes('.expert.'));

  if (expertGroups.length > 0) {
    let totalSize = 0;
    for (const [, group] of expertGroups) {
      for (const shardIdx of group.shards) {
        const shard = manifest?.shards[shardIdx];
        if (shard) totalSize += shard.size;
      }
    }
    return Math.floor(totalSize / expertGroups.length);
  }

  return manifest?.moeConfig?.expertBytes || 0;
}

export function getLayerGroupIds() {
  const ids = Object.keys(getManifest()?.groups || {})
    .filter(id => id.startsWith('layer.'));
  return sortGroupIds(ids);
}

export function getExpertGroupIds(layerIdx) {
  const prefix = `layer.${layerIdx}.expert.`;
  return Object.keys(getManifest()?.groups || {})
    .filter(id => id.startsWith(prefix))
    .sort((a, b) => {
      const expertA = parseGroupExpertIndex(a) ?? 0;
      const expertB = parseGroupExpertIndex(b) ?? 0;
      return expertA - expertB;
    });
}
