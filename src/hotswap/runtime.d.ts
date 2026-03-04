import type { HotSwapManifest } from './manifest.js';
import type { HotSwapConfigSchema } from '../config/schema/hotswap.schema.js';

export interface HotSwapRolloutContext {
  subjectId?: string | null;
  modelId?: string | null;
  modelUrl?: string | null;
  sessionId?: string | null;
  optInTag?: string | null;
  forceEnable?: boolean;
}

export interface HotSwapRolloutDecision {
  allowed: boolean;
  mode: string;
  reason: string;
  bucket: number | null;
  threshold: number | null;
  subjectId: string | null;
}

export declare function evaluateHotSwapRollout(
  policy: HotSwapConfigSchema | null | undefined,
  context?: HotSwapRolloutContext
): HotSwapRolloutDecision;

export declare function getHotSwapManifest(): HotSwapManifest | null;

export declare function setHotSwapManifest(manifest: HotSwapManifest | null): void;

export declare function getLastHotSwapRolloutDecision(): HotSwapRolloutDecision | null;
