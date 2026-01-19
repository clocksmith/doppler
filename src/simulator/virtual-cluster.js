
import { log } from '../debug/index.js';
import { MODULE } from './virtual-utils.js';
import { VirtualGPU } from './virtual-gpu.js';
import { VirtualCPU } from './virtual-cpu.js';

export class VirtualCluster {
  constructor(config) {
    this.config = config;

    this.gpus = [];

    this.cpus = [];

    // Create virtual GPUs
    for (let i = 0; i < config.topology.gpuCount; i++) {
      this.gpus.push(new VirtualGPU(i, config.gpuSpec, config.opfsRootPath));
    }

    // Create virtual CPUs
    for (let i = 0; i < config.topology.cpuCount; i++) {
      this.cpus.push(new VirtualCPU(i, config.cpuSpec, config.opfsRootPath));
    }

    log.info(MODULE, `VirtualCluster created: ${config.topology.gpuCount} GPUs, ${config.topology.cpuCount} CPUs`);
  }

  async initialize() {
    const promises = [
    ...this.gpus.map(gpu => gpu.initialize()),
    ...this.cpus.map(cpu => cpu.initialize()),
    ];
    await Promise.all(promises);
    log.info(MODULE, `VirtualCluster initialized`);
  }

  getGPU(index) {
    if (index < 0 || index >= this.gpus.length) {
      throw new Error(`GPU index ${index} out of range (0-${this.gpus.length - 1})`);
    }
    return this.gpus[index];
  }

  getCPU(index) {
    if (index < 0 || index >= this.cpus.length) {
      throw new Error(`CPU index ${index} out of range (0-${this.cpus.length - 1})`);
    }
    return this.cpus[index];
  }

  getNodeGPUs(nodeIndex) {
    const gpusPerNode = this.config.topology.gpusPerNode;
    const startIdx = nodeIndex * gpusPerNode;
    const endIdx = Math.min(startIdx + gpusPerNode, this.gpus.length);
    return this.gpus.slice(startIdx, endIdx);
  }

  async allocateOnGPU(gpuIndex, sizeBytes, label) {
    return this.getGPU(gpuIndex).allocate(sizeBytes, label);
  }

  async allocateSharded(gpuIndices, totalSizeBytes, label) {
    const shardSize = Math.ceil(totalSizeBytes / gpuIndices.length);
    const refs = await Promise.all(
    gpuIndices.map((gpuIdx, i) =>
    this.getGPU(gpuIdx).allocate(shardSize, `${label || 'shard'}_${i}`)
    )
    );
    return refs;
  }

  getClusterStats() {
    const gpuStats = this.gpus.map(gpu => gpu.getMemoryStats());
    const cpuStats = this.cpus.map(cpu => cpu.getMemoryStats());

    let totalVram = 0;
    let totalCpuMem = 0;
    let totalAllocated = 0;
    let vramUsed = 0;
    let ramUsed = 0;
    let opfsUsed = 0;

    for (const stat of gpuStats) {
      totalVram += stat.totalVramBytes;
      totalAllocated += stat.allocatedBytes;
      vramUsed += stat.vramUsedBytes;
      ramUsed += stat.ramUsedBytes;
      opfsUsed += stat.opfsUsedBytes;
    }

    for (const stat of cpuStats) {
      totalCpuMem += stat.totalMemoryBytes;
      totalAllocated += stat.allocatedBytes;
    }

    return {
      gpuCount: this.gpus.length,
      cpuCount: this.cpus.length,
      totalVramBytes: totalVram,
      totalCpuMemoryBytes: totalCpuMem,
      totalAllocatedBytes: totalAllocated,
      vramUsedBytes: vramUsed,
      ramUsedBytes: ramUsed,
      opfsUsedBytes: opfsUsed,
      gpuStats,
      cpuStats,
    };
  }

  async destroy() {
    const promises = [
    ...this.gpus.map(gpu => gpu.destroy()),
    ...this.cpus.map(cpu => cpu.destroy()),
    ];
    await Promise.all(promises);
    log.info(MODULE, `VirtualCluster destroyed`);
  }
}

export function createVirtualCluster(config) {
  return new VirtualCluster(config);
}
