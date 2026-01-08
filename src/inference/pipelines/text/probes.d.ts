/**
 * Config-driven probe helpers for targeted GPU buffer inspection.
 *
 * Probes let us read specific token/dimension values without adding
 * ad-hoc logs in the code. Probes are configured via runtime.shared.debug.probes.
 */

import type { ProbeConfigSchema, ProbeStage } from '../../../config/schema/index.js';
import type { CommandRecorder } from '../../../gpu/command-recorder.js';

/**
 * Run configured probes for a specific stage.
 */
export function runProbes(
  stage: ProbeStage,
  buffer: GPUBuffer | Float32Array,
  options: {
    layerIdx?: number;
    numTokens: number;
    hiddenSize: number;
    probes?: ProbeConfigSchema[] | null;
    recorder?: CommandRecorder | null;
    dtype?: 'f16' | 'f32';
  }
): Promise<void>;

export function hasProbeStage(
  probes: ProbeConfigSchema[] | null | undefined,
  stage: ProbeStage,
  layerIdx?: number
): boolean;
