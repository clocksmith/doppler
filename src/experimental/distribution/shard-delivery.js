import { downloadDistributedShard } from '../../storage/distribution-transport.js';
import { normalizeP2PConfig } from './shard-delivery/retry.js';
import { downloadShardFromP2P } from './shard-delivery/receipt.js';
import { P2P_TRANSPORT_ERROR_CODES } from './p2p-transport-contract.js';

export { resolveShardDeliveryPlan, getSourceOrder, getInFlightShardDeliveryCount } from '../../storage/distribution-transport.js';

// Optional peer assembly stays above byte storage. The shared delivery policy
// owns source transitions, integrity checks, cancellation, and observations.
export function downloadShard(baseUrl, shardIndex, shardInfo, options = {}) {
  return downloadDistributedShard(baseUrl, shardIndex, shardInfo, {
    ...options,
    peerDelivery(config) {
      const peer = normalizeP2PConfig(config);
      return {
        enabled: peer.enabled,
        transport: peer.transport,
        download: (index, info, request) => downloadShardFromP2P(index, info, peer, request),
        isMiss: error => error?.code === P2P_TRANSPORT_ERROR_CODES.unconfigured
          || error?.code === P2P_TRANSPORT_ERROR_CODES.unavailable,
      };
    },
  });
}
