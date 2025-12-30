export type ParamCategoryName = 'generation' | 'model' | 'session' | 'hybrid';

export const ParamCategory: {
  readonly GENERATION: 'generation';
  readonly MODEL: 'model';
  readonly SESSION: 'session';
  readonly HYBRID: 'hybrid';
};

export const PARAM_CATEGORIES: Record<string, ParamCategoryName>;

export const CategoryRules: Record<
  ParamCategoryName,
  { callTime: boolean; runtime: boolean; manifest: boolean }
>;

export function getParamCategory(name: string): ParamCategoryName | null;
