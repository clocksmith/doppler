export interface ToolRegistryEntry {
  id: string;
  script: string;
  description: string;
  configKey: string | null;
}

export declare const TOOL_REGISTRY: Record<string, ToolRegistryEntry>;

export declare function listTools(): string[];
export declare function getTool(id: string): ToolRegistryEntry | null;
