/**
 * Virtual Device Layer for NVIDIA Superchip Simulation
 *
 * Provides VirtualGPU and VirtualCluster abstractions that map
 * emulated GPU resources to local VRAM, RAM, and OPFS storage.
 *
 * @module simulator/virtual-device
 */

// Re-export all components for backward compatibility
export { generateBufferId } from './virtual-utils.js';
export { VirtualGPU } from './virtual-gpu.js';
export { VirtualCPU } from './virtual-cpu.js';
export { VirtualCluster, createVirtualCluster } from './virtual-cluster.js';
