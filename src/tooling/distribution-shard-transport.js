let distributionModulePromise = null;

async function getDistributionModule() {
  distributionModulePromise ??= import('../experimental/distribution/shard-delivery.js');
  return distributionModulePromise;
}

export async function downloadShardWithOptionalDistribution(baseUrl, shardIndex, shardInfo, options = {}) {
  const { downloadShard } = await getDistributionModule();
  return downloadShard(baseUrl, shardIndex, shardInfo, options);
}
