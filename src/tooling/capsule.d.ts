import type { DopplerCapsule, CapsuleIdentity } from '../config/capsule.js';
export declare function loadCapsule(path: string, options?: { signal?: AbortSignal | null }): Promise<DopplerCapsule>;
export declare function writeCapsule(path: string, capsule: DopplerCapsule): Promise<CapsuleIdentity & { outputPath: string }>;
