
import { log } from '../debug/index.js';

// =============================================================================
// Constants
// =============================================================================

const MODULE = 'NVLink-C2C';

let transferIdCounter = 0;

// =============================================================================
// NVLink-C2C Controller
// =============================================================================

export class NVLinkC2CController {
  constructor(spec, timingModel) {
    this.spec = spec;

    this.timingModel = timingModel;

    this._totalTransfers = 0;

    this._totalBytesTransferred = 0;

    this._totalSimulatedTimeMs = 0;

    this._totalActualTimeMs = 0;

    this._cpuToGpuTransfers = 0;

    this._gpuToCpuTransfers = 0;

    this._coherentReads = 0;

    this._coherentWrites = 0;

    log.verbose(MODULE, `Initialized: bandwidth=${this.spec.bandwidthBytesPerSec / 1e9} GB/s, coherent=${this.spec.coherent}`);
  }

  async cpuToGpu(cpu, cpuBufferId, gpu, gpuBufferId, sizeBytes) {
    const start = performance.now();
    const transferId = `c2c_${Date.now()}_${transferIdCounter++}`;

    // Read from CPU
    const cpuData = await cpu.read(cpuBufferId);
    const transferSize = sizeBytes ?? cpuData.byteLength;
    const dataToTransfer = sizeBytes ? cpuData.slice(0, sizeBytes) : cpuData;

    // Calculate simulated transfer time
    const timing = this.timingModel.computeNvlinkTimeMs(
    transferSize,
    -1, // CPU
    gpu.index,
    this.spec.bandwidthBytesPerSec
    );

    // Inject delay
    await this.timingModel.injectDelay(timing.timeMs);

    // Write to GPU
    let targetBufferId = gpuBufferId;
    if (!targetBufferId) {
      const bufRef = await gpu.allocate(transferSize, `c2c_from_cpu${cpu.index}`);
      targetBufferId = bufRef.metadata.id;
    }
    await gpu.write(targetBufferId, dataToTransfer);

    const actualTimeMs = performance.now() - start;

    // Update stats
    this._totalTransfers++;
    this._totalBytesTransferred += transferSize;
    this._totalSimulatedTimeMs += timing.timeMs;
    this._totalActualTimeMs += actualTimeMs;
    this._cpuToGpuTransfers++;

    log.verbose(MODULE, `CPU${cpu.index}→GPU${gpu.index}: ${transferSize} bytes, ${timing.timeMs.toFixed(3)}ms sim, ${actualTimeMs.toFixed(3)}ms actual`);

    return {
      transferId,
      bytesTransferred: transferSize,
      direction: 'cpu_to_gpu',
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
      source: { type: 'cpu', index: cpu.index },
      destination: { type: 'gpu', index: gpu.index },
    };
  }

  async gpuToCpu(gpu, gpuBufferId, cpu, cpuBufferId, sizeBytes) {
    const start = performance.now();
    const transferId = `c2c_${Date.now()}_${transferIdCounter++}`;

    // Read from GPU
    const gpuData = await gpu.read(gpuBufferId);
    const transferSize = sizeBytes ?? gpuData.byteLength;
    const dataToTransfer = sizeBytes ? gpuData.slice(0, sizeBytes) : gpuData;

    // Calculate simulated transfer time
    const timing = this.timingModel.computeNvlinkTimeMs(
    transferSize,
    gpu.index,
    -1, // CPU
    this.spec.bandwidthBytesPerSec
    );

    // Inject delay
    await this.timingModel.injectDelay(timing.timeMs);

    // Write to CPU
    let targetBufferId = cpuBufferId;
    if (!targetBufferId) {
      const bufRef = await cpu.allocate(transferSize, `c2c_from_gpu${gpu.index}`);
      targetBufferId = bufRef.metadata.id;
    }
    await cpu.write(targetBufferId, dataToTransfer);

    const actualTimeMs = performance.now() - start;

    // Update stats
    this._totalTransfers++;
    this._totalBytesTransferred += transferSize;
    this._totalSimulatedTimeMs += timing.timeMs;
    this._totalActualTimeMs += actualTimeMs;
    this._gpuToCpuTransfers++;

    log.verbose(MODULE, `GPU${gpu.index}→CPU${cpu.index}: ${transferSize} bytes, ${timing.timeMs.toFixed(3)}ms sim, ${actualTimeMs.toFixed(3)}ms actual`);

    return {
      transferId,
      bytesTransferred: transferSize,
      direction: 'gpu_to_cpu',
      simulatedTimeMs: timing.timeMs,
      actualTimeMs,
      source: { type: 'gpu', index: gpu.index },
      destination: { type: 'cpu', index: cpu.index },
    };
  }

  async coherentRead(cpu, cpuBufferId, offset = 0, length) {
    const start = performance.now();
    const transferId = `c2c_coh_read_${Date.now()}_${transferIdCounter++}`;

    // Read from CPU memory
    const fullData = await cpu.read(cpuBufferId);
    const readLength = length ?? fullData.byteLength - offset;
    const data = fullData.slice(offset, offset + readLength);

    // Calculate transfer time with C2C latency
    const timing = this.timingModel.computeNvlinkTimeMs(
    readLength,
    -1, // CPU
    0,  // Any GPU
    this.spec.bandwidthBytesPerSec
    );

    // Add coherency latency
    const coherencyLatencyMs = this.spec.latencyUs / 1000;
    const totalSimulatedMs = timing.timeMs + coherencyLatencyMs;

    await this.timingModel.injectDelay(totalSimulatedMs);

    const actualTimeMs = performance.now() - start;

    // Update stats
    this._totalTransfers++;
    this._totalBytesTransferred += readLength;
    this._totalSimulatedTimeMs += totalSimulatedMs;
    this._totalActualTimeMs += actualTimeMs;
    this._coherentReads++;

    log.verbose(MODULE, `Coherent read from CPU${cpu.index}: ${readLength} bytes, ${totalSimulatedMs.toFixed(3)}ms sim`);

    return {
      data,
      transferResult: {
        transferId,
        bytesTransferred: readLength,
        direction: 'cpu_to_gpu',
        simulatedTimeMs: totalSimulatedMs,
        actualTimeMs,
        source: { type: 'cpu', index: cpu.index },
        destination: { type: 'gpu', index: 0 },
      },
    };
  }

  async coherentWrite(cpu, cpuBufferId, data, offset = 0) {
    const start = performance.now();
    const transferId = `c2c_coh_write_${Date.now()}_${transferIdCounter++}`;

    // Calculate transfer time with C2C latency
    const timing = this.timingModel.computeNvlinkTimeMs(
    data.byteLength,
    0,  // Any GPU
    -1, // CPU
    this.spec.bandwidthBytesPerSec
    );

    const coherencyLatencyMs = this.spec.latencyUs / 1000;
    const totalSimulatedMs = timing.timeMs + coherencyLatencyMs;

    await this.timingModel.injectDelay(totalSimulatedMs);

    // Write to CPU memory
    const existingData = await cpu.read(cpuBufferId);
    const updated = new Uint8Array(existingData);
    updated.set(new Uint8Array(data), offset);
    await cpu.write(cpuBufferId, updated.buffer);

    const actualTimeMs = performance.now() - start;

    // Update stats
    this._totalTransfers++;
    this._totalBytesTransferred += data.byteLength;
    this._totalSimulatedTimeMs += totalSimulatedMs;
    this._totalActualTimeMs += actualTimeMs;
    this._coherentWrites++;

    log.verbose(MODULE, `Coherent write to CPU${cpu.index}: ${data.byteLength} bytes, ${totalSimulatedMs.toFixed(3)}ms sim`);

    return {
      transferId,
      bytesTransferred: data.byteLength,
      direction: 'gpu_to_cpu',
      simulatedTimeMs: totalSimulatedMs,
      actualTimeMs,
      source: { type: 'gpu', index: 0 },
      destination: { type: 'cpu', index: cpu.index },
    };
  }

  getStats() {
    return {
      totalTransfers: this._totalTransfers,
      totalBytesTransferred: this._totalBytesTransferred,
      totalSimulatedTimeMs: this._totalSimulatedTimeMs,
      totalActualTimeMs: this._totalActualTimeMs,
      cpuToGpuTransfers: this._cpuToGpuTransfers,
      gpuToCpuTransfers: this._gpuToCpuTransfers,
      coherentReads: this._coherentReads,
      coherentWrites: this._coherentWrites,
    };
  }

  resetStats() {
    this._totalTransfers = 0;
    this._totalBytesTransferred = 0;
    this._totalSimulatedTimeMs = 0;
    this._totalActualTimeMs = 0;
    this._cpuToGpuTransfers = 0;
    this._gpuToCpuTransfers = 0;
    this._coherentReads = 0;
    this._coherentWrites = 0;
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createNVLinkC2CController(spec, timingModel) {
  return new NVLinkC2CController(spec, timingModel);
}
