import type { InferencePipeline } from '../../inference/pipeline.js';
import type { RDRRManifest } from '../../storage/rdrr-format.js';
import type { LoRAManifest } from '../../adapters/lora-loader.js';
import type {
  TextModelConfig,
  LoadProgressEvent,
} from './types.js';

export declare function getPipeline(): InferencePipeline | null;

export declare function getCurrentModelId(): string | null;

export declare function extractTextModelConfig(manifest: RDRRManifest): TextModelConfig;

export declare function readOPFSFile(path: string): Promise<ArrayBuffer>;

export declare function writeOPFSFile(path: string, data: ArrayBuffer): Promise<void>;

export declare function fetchArrayBuffer(url: string): Promise<ArrayBuffer>;

export declare function initDoppler(): Promise<boolean>;

export declare function loadModel(
  modelId: string,
  modelUrl?: string | null,
  onProgress?: ((event: LoadProgressEvent) => void) | null,
  localPath?: string | null
): Promise<boolean>;

export declare function unloadModel(): Promise<void>;

export declare function loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void>;

export declare function unloadLoRAAdapter(): Promise<void>;

export declare function getActiveLoRA(): string | null;

export declare function getAvailableModels(): Promise<string[]>;

export declare function getDopplerStorageInfo(): Promise<unknown>;

export declare function destroyDoppler(): Promise<void>;
