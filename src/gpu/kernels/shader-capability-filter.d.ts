export declare function shaderEnablesSubgroups(source: string): boolean;

export declare function filterShaderForCapabilities(
  source: string,
  capabilities: { hasSubgroups: boolean; label?: string }
): { source: string; filtered: boolean; removedFunctions: string[] };
