import type { DopplerPack, PackIdentity } from '../config/pack.js';
export declare function loadPack(path: string): Promise<DopplerPack>;
export declare function writePack(path: string, pack: DopplerPack): Promise<PackIdentity & { outputPath: string }>;
