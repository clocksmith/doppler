export interface ParameterModel {
  paramGroups?: () => Record<string, unknown>;
  loraParams?: () => unknown[];
}
export declare function resolveModelParamGroups(model: ParameterModel | null | undefined): Record<string, unknown>;
export declare function selectTrainableParamGroups(paramGroups: Record<string, unknown>, freezeMap?: Record<string, boolean> | null): { trainableGroups: Record<string, unknown[]>; frozenGroups: string[] };
export declare function isTensorLike(value: unknown): boolean;
