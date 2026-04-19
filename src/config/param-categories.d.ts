export type ParamCategoryName = 'generation' | 'model' | 'session' | 'hybrid';

export const PARAM_CATEGORIES: Record<string, ParamCategoryName>;

export const CategoryRules: Record<
  ParamCategoryName,
  { callTime: boolean; runtime: boolean; manifest: boolean }
>;
