import { downloadShardWithOptionalDistribution } from '../tooling/distribution-shard-transport.js';

export async function downloadDistributedShard(baseUrl, shardIndex, shardInfo, options = {}) {
  return downloadShardWithOptionalDistribution(baseUrl, shardIndex, shardInfo, options);
}
