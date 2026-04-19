

import { getManifest } from './parsing.js';

export function getShardsForExpert(layerIdx, expertIdx, manifest = getManifest()) {
  const groupId = `layer.${layerIdx}.expert.${expertIdx}`;
  const group = manifest?.groups?.[groupId];
  if (group) {
    return group.shards;
  }
  throw new Error(`Missing expert group mapping: ${groupId}`);
}

export function getTensorsForExpert(layerIdx, expertIdx, manifest = getManifest()) {
  const groupId = `layer.${layerIdx}.expert.${expertIdx}`;
  const group = manifest?.groups?.[groupId];
  if (group) {
    return group.tensors;
  }
  throw new Error(`Missing expert group mapping: ${groupId}`);
}

export function getExpertBytes(manifest = getManifest()) {
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
