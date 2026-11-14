export interface Qwen4ExpGGUFConfig extends Record<string, unknown> {
  num_hidden_layers: number;
  layer_types: string[];
  hidden_size: number;
  vocab_size: number;
}
export interface Qwen4ExpGGUFTable { name: string; dtype: string; shape: number[]; size: number }
export declare function normalizeQwen4ExpGGUFConfig(config: Qwen4ExpGGUFConfig, metadata: Record<string, unknown>, family: { outputGateType?: string; indexerKvHeads?: number }): void;
export declare function bindQwen4ExpGGUFTable(config: Record<string, unknown>, tensors: Qwen4ExpGGUFTable[]): { table: Qwen4ExpGGUFTable; name: string };
