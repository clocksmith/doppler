/**
 * LoRA configuration schema.
 */

export interface LoraConfigSchema {
  rank: number;
  alpha: number;
  dropout: number;
  targetModules: string[];
}

export declare const DEFAULT_LORA_CONFIG: LoraConfigSchema;
