
import { log } from '../debug/index.js';

// =============================================================================
// Constants
// =============================================================================

const MODULE = 'NVLinkFabric';

const CROSS_NODE_BANDWIDTH_FACTOR = 0.8;

let operationIdCounter = 0;

// =============================================================================
// NVLink Fabric Controller
// =============================================================================

export class NVLinkFabric {
  constructor(spec, topology, timingModel, cluster) {
    this.spec = spec;

    this.topology = topology;

    this.timingModel = timingModel;

    this._cluster = cluster;

    // Stats
    this._totalP2PTransfers = 0;
    this._totalCollectives = 0;
    this._totalBytesTransferred = 0;
    this._totalSimulatedTimeMs = 0;
    this._totalActualTimeMs = 0;
    this._collectiveBreakdown = new Map();

    log.verbose(MODULE, `Initialized: ${topology.gpuCount} GPUs, ${topology.nodeCount} nodes, ${spec.bandwidthBytesPerSec / 1e9} GB/s`);
  }

  sameNode(gpu1, gpu2) {
    const node1 = Math.floor(gpu1 / this.topology.gpusPerNode);
    const node2 = Math.floor(gpu2 / this.topology.gpusPerNode);
    return node1 === node2;
  }

  getEffectiveBandwidth(gpu1, gpu2) {
    if (this.sameNode(gpu1, gpu2)) {
      return this.spec.bandwidthBytesPerSec;
    }
    // Cross-node has lower effective bandwidth
    return this.spec.bandwidthBytesPerSec * CROSS_NODE_BANDWIDTH_FACTOR;
  }

  async send(srcGpu, srcBufferId, dstGpu, dstBufferId, sizeBytes) {
    const start = performance.now();
    const transferId = `nvlink_p2p_${Date.now()}_${operationIdCounter++}`;

    const srcGpuDev = this._cluster.getGPU(srcGpu);
    const dstGpuDev = this._cluster.getGPU(dstGpu);

    // Read from source
    const data = await srcGpuDev.read(srcBufferId);
    const transferSize = sizeBytes ?? data.byteLength;
    const dataToTransfer = sizeBytes ? data.slice(0, sizeBytes) : data;

    // Calculate timing
    const bandwidth = this.getEffectiveBandwidth(srcGpu, dstGpu);
    const timing = this.timingModel.computeNvlinkTimeMs(transferSize, srcGpu, dstGpu, bandwidth);

    await this.timingModel.injectDelay(timing.timeMs);

    // Write to destination
    let targetId = dstBufferId;
    if (!targetId) {
      const bufRef = await dstGpuDev.allocate(transferSize, `from_gpu${srcGpu}`);
      targetId = bufRef.metadata.id;
    }
    await dstGpuDev.write(targetId, dataToTransfer);

    const actualTimeMs = performance.now() - start;

    // Update stats
    this._totalP2PTransfers++;
    this._totalBytesTransferred += transferSize;
    this._totalSimulatedTimeMs += timing.timeMs;
    this._totalActualTimeMs += actualTimeMs;

    log.verbose(MODULE, `P2P GPU${srcGpu}→GPU${dstGpu}: ${transferSize} bytes, ${timing.timeMs.toFixed(3)}ms sim`);

    return {
      transferId,
      srcGpu,
      dstGpu,
      bytesTransferred: transferSize,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  async copy(srcGpu, srcBufferId, dstGpu, label) {
    const srcGpuDev = this._cluster.getGPU(srcGpu);
    const dstGpuDev = this._cluster.getGPU(dstGpu);

    const srcInfo = srcGpuDev.getBufferInfo(srcBufferId);
    if (!srcInfo) {
      throw new Error(`Buffer ${srcBufferId} not found on GPU ${srcGpu}`);
    }

    const bufferRef = await dstGpuDev.allocate(srcInfo.sizeBytes, label || `copy_from_gpu${srcGpu}`);
    const transfer = await this.send(srcGpu, srcBufferId, dstGpu, bufferRef.metadata.id);

    return { bufferRef, transfer };
  }

  async allReduce(bufferIds, gpuIndices) {
    const start = performance.now();
    const operationId = `all_reduce_${Date.now()}_${operationIdCounter++}`;

    // Get buffer size from first GPU
    const firstGpu = gpuIndices[0];
    const firstBufferId = bufferIds.get(firstGpu);
    const firstGpuDev = this._cluster.getGPU(firstGpu);
    const bufferInfo = firstGpuDev.getBufferInfo(firstBufferId);
    const bufferSize = bufferInfo.sizeBytes;

    // Calculate timing for ring all-reduce
    const timing = this.timingModel.computeAllReduceTimeMs(bufferSize, gpuIndices.length);

    await this.timingModel.injectDelay(timing.timeMs);

    // Simulate the reduction: read all buffers, sum, write back
    const buffers = await Promise.all(
    gpuIndices.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      return gpu.read(bufferIds.get(gpuIdx));
    })
    );

    // Sum all buffers (assuming F32 for simplicity)
    const result = new Float32Array(bufferSize / 4);
    for (const buf of buffers) {
      const arr = new Float32Array(buf);
      for (let i = 0; i < result.length; i++) {
        result[i] += arr[i];
      }
    }

    // Write result back to all GPUs
    await Promise.all(
    gpuIndices.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      await gpu.write(bufferIds.get(gpuIdx), result.buffer);
    })
    );

    const actualTimeMs = performance.now() - start;
    const totalBytes = bufferSize * 2 * (gpuIndices.length - 1); // Ring all-reduce transfers

    this._updateCollectiveStats('all_reduce', totalBytes, timing.timeMs, actualTimeMs);

    log.verbose(MODULE, `All-reduce on GPUs [${gpuIndices.join(',')}]: ${bufferSize} bytes, ${timing.timeMs.toFixed(3)}ms sim`);

    return {
      operationId,
      type: 'all_reduce',
      gpuIndices,
      totalBytesTransferred: totalBytes,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  async allGather(bufferIds, gpuIndices) {
    const start = performance.now();
    const operationId = `all_gather_${Date.now()}_${operationIdCounter++}`;

    // Get per-GPU buffer size
    const firstGpu = gpuIndices[0];
    const firstBufferId = bufferIds.get(firstGpu);
    const firstGpuDev = this._cluster.getGPU(firstGpu);
    const bufferInfo = firstGpuDev.getBufferInfo(firstBufferId);
    const perGpuSize = bufferInfo.sizeBytes;

    // Calculate timing
    const timing = this.timingModel.computeAllGatherTimeMs(perGpuSize, gpuIndices.length);

    await this.timingModel.injectDelay(timing.timeMs);

    // Read all buffers
    const buffers = await Promise.all(
    gpuIndices.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      return gpu.read(bufferIds.get(gpuIdx));
    })
    );

    // Concatenate all buffers
    const totalSize = perGpuSize * gpuIndices.length;
    const gathered = new Uint8Array(totalSize);
    let offset = 0;
    for (const buf of buffers) {
      gathered.set(new Uint8Array(buf), offset);
      offset += buf.byteLength;
    }

    // Allocate new buffers on each GPU to hold gathered result
    // In real all-gather, each GPU ends up with the full concatenated result
    await Promise.all(
    gpuIndices.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      const resultBuf = await gpu.allocate(totalSize, `all_gather_result`);
      await gpu.write(resultBuf.metadata.id, gathered.buffer);
    })
    );

    const actualTimeMs = performance.now() - start;
    const totalBytes = totalSize * (gpuIndices.length - 1);

    this._updateCollectiveStats('all_gather', totalBytes, timing.timeMs, actualTimeMs);

    log.verbose(MODULE, `All-gather on GPUs [${gpuIndices.join(',')}]: ${perGpuSize} bytes/GPU, ${timing.timeMs.toFixed(3)}ms sim`);

    return {
      operationId,
      type: 'all_gather',
      gpuIndices,
      totalBytesTransferred: totalBytes,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  async reduceScatter(bufferIds, gpuIndices) {
    const start = performance.now();
    const operationId = `reduce_scatter_${Date.now()}_${operationIdCounter++}`;

    // Get buffer size
    const firstGpu = gpuIndices[0];
    const firstGpuDev = this._cluster.getGPU(firstGpu);
    const bufferInfo = firstGpuDev.getBufferInfo(bufferIds.get(firstGpu));
    const totalSize = bufferInfo.sizeBytes;
    const shardSize = totalSize / gpuIndices.length;

    // Calculate timing
    const timing = this.timingModel.computeReduceScatterTimeMs(totalSize, gpuIndices.length);

    await this.timingModel.injectDelay(timing.timeMs);

    // Read all buffers and reduce
    const buffers = await Promise.all(
    gpuIndices.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      return gpu.read(bufferIds.get(gpuIdx));
    })
    );

    const reduced = new Float32Array(totalSize / 4);
    for (const buf of buffers) {
      const arr = new Float32Array(buf);
      for (let i = 0; i < reduced.length; i++) {
        reduced[i] += arr[i];
      }
    }

    // Scatter shards to each GPU
    await Promise.all(
    gpuIndices.map(async (gpuIdx, i) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      const shardStart = i * (shardSize / 4);
      const shardEnd = shardStart + (shardSize / 4);
      const shard = reduced.slice(shardStart, shardEnd);
      const shardBuf = await gpu.allocate(shardSize, `reduce_scatter_shard`);
      await gpu.write(shardBuf.metadata.id, shard.buffer);
    })
    );

    const actualTimeMs = performance.now() - start;
    const totalBytes = totalSize * (gpuIndices.length - 1) / gpuIndices.length;

    this._updateCollectiveStats('reduce_scatter', totalBytes, timing.timeMs, actualTimeMs);

    log.verbose(MODULE, `Reduce-scatter on GPUs [${gpuIndices.join(',')}]: ${totalSize} bytes total, ${timing.timeMs.toFixed(3)}ms sim`);

    return {
      operationId,
      type: 'reduce_scatter',
      gpuIndices,
      totalBytesTransferred: totalBytes,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  async broadcast(srcGpu, srcBufferId, gpuIndices) {
    const start = performance.now();
    const operationId = `broadcast_${Date.now()}_${operationIdCounter++}`;

    const srcGpuDev = this._cluster.getGPU(srcGpu);
    const data = await srcGpuDev.read(srcBufferId);
    const bufferSize = data.byteLength;

    // Calculate timing (tree broadcast: log2(n) steps)
    const steps = Math.ceil(Math.log2(gpuIndices.length));
    const timing = this.timingModel.computeNvlinkTimeMs(bufferSize * steps, srcGpu, gpuIndices[1] || srcGpu);

    await this.timingModel.injectDelay(timing.timeMs);

    // Write to all destination GPUs
    const dstGpus = gpuIndices.filter(g => g !== srcGpu);
    await Promise.all(
    dstGpus.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      const bufRef = await gpu.allocate(bufferSize, `broadcast_from_gpu${srcGpu}`);
      await gpu.write(bufRef.metadata.id, data);
    })
    );

    const actualTimeMs = performance.now() - start;
    const totalBytes = bufferSize * dstGpus.length;

    this._updateCollectiveStats('broadcast', totalBytes, timing.timeMs, actualTimeMs);

    log.verbose(MODULE, `Broadcast from GPU${srcGpu} to [${dstGpus.join(',')}]: ${bufferSize} bytes, ${timing.timeMs.toFixed(3)}ms sim`);

    return {
      operationId,
      type: 'broadcast',
      gpuIndices,
      totalBytesTransferred: totalBytes,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  async scatter(srcGpu, srcBufferId, gpuIndices) {
    const start = performance.now();
    const operationId = `scatter_${Date.now()}_${operationIdCounter++}`;

    const srcGpuDev = this._cluster.getGPU(srcGpu);
    const data = await srcGpuDev.read(srcBufferId);
    const totalSize = data.byteLength;
    const chunkSize = totalSize / gpuIndices.length;

    // Calculate timing
    const timing = this.timingModel.computeNvlinkTimeMs(totalSize - chunkSize, srcGpu, gpuIndices[1] || srcGpu);

    await this.timingModel.injectDelay(timing.timeMs);

    // Distribute chunks
    await Promise.all(
    gpuIndices.map(async (gpuIdx, i) => {
      const chunk = data.slice(i * chunkSize, (i + 1) * chunkSize);
      if (gpuIdx === srcGpu) {
        // Source GPU keeps its chunk
        return;
      }
      const gpu = this._cluster.getGPU(gpuIdx);
      const bufRef = await gpu.allocate(chunkSize, `scatter_chunk_${i}`);
      await gpu.write(bufRef.metadata.id, chunk);
    })
    );

    const actualTimeMs = performance.now() - start;
    const totalBytes = totalSize - chunkSize;

    this._updateCollectiveStats('scatter', totalBytes, timing.timeMs, actualTimeMs);

    return {
      operationId,
      type: 'send', // scatter is like multiple sends
      gpuIndices,
      totalBytesTransferred: totalBytes,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  async gather(bufferIds, dstGpu) {
    const start = performance.now();
    const operationId = `gather_${Date.now()}_${operationIdCounter++}`;

    const gpuIndices = Array.from(bufferIds.keys());

    // Read all chunks
    const chunks = await Promise.all(
    gpuIndices.map(async (gpuIdx) => {
      const gpu = this._cluster.getGPU(gpuIdx);
      return { gpuIdx, data: await gpu.read(bufferIds.get(gpuIdx)) };
    })
    );

    // Sort by GPU index for consistent ordering
    chunks.sort((a, b) => a.gpuIdx - b.gpuIdx);

    const totalSize = chunks.reduce((sum, c) => sum + c.data.byteLength, 0);

    // Calculate timing
    const bytesFromOthers = totalSize - chunks.find(c => c.gpuIdx === dstGpu)?.data.byteLength || 0;
    const timing = this.timingModel.computeNvlinkTimeMs(bytesFromOthers, gpuIndices[0], dstGpu);

    await this.timingModel.injectDelay(timing.timeMs);

    // Concatenate on destination GPU
    const gathered = new Uint8Array(totalSize);
    let offset = 0;
    for (const { data } of chunks) {
      gathered.set(new Uint8Array(data), offset);
      offset += data.byteLength;
    }

    const dstGpuDev = this._cluster.getGPU(dstGpu);
    const resultBuf = await dstGpuDev.allocate(totalSize, `gather_result`);
    await dstGpuDev.write(resultBuf.metadata.id, gathered.buffer);

    const actualTimeMs = performance.now() - start;

    this._updateCollectiveStats('gather', bytesFromOthers, timing.timeMs, actualTimeMs);

    return {
      operationId,
      type: 'recv', // gather is like multiple receives
      gpuIndices,
      totalBytesTransferred: bytesFromOthers,
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
    };
  }

  _updateCollectiveStats(type, bytes, simTimeMs, actualTimeMs) {
    this._totalCollectives++;
    this._totalBytesTransferred += bytes;
    this._totalSimulatedTimeMs += simTimeMs;
    this._totalActualTimeMs += actualTimeMs;

    if (!this._collectiveBreakdown.has(type)) {
      this._collectiveBreakdown.set(type, { count: 0, bytes: 0, timeMs: 0 });
    }
    const stats = this._collectiveBreakdown.get(type);
    stats.count++;
    stats.bytes += bytes;
    stats.timeMs += simTimeMs;
  }

  getStats() {
    return {
      totalP2PTransfers: this._totalP2PTransfers,
      totalCollectives: this._totalCollectives,
      totalBytesTransferred: this._totalBytesTransferred,
      totalSimulatedTimeMs: this._totalSimulatedTimeMs,
      totalActualTimeMs: this._totalActualTimeMs,
      collectiveBreakdown: new Map(this._collectiveBreakdown),
    };
  }

  resetStats() {
    this._totalP2PTransfers = 0;
    this._totalCollectives = 0;
    this._totalBytesTransferred = 0;
    this._totalSimulatedTimeMs = 0;
    this._totalActualTimeMs = 0;
    this._collectiveBreakdown.clear();
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createNVLinkFabric(spec, topology, timingModel, cluster) {
  return new NVLinkFabric(spec, topology, timingModel, cluster);
}
