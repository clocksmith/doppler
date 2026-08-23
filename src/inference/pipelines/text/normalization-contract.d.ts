export function resolvePostNormContract(normalization: {
  rmsNormEps: number;
  rmsNormWeightOffset: boolean;
  postNormEps?: number | null;
  postNormWeightOffset?: boolean | null;
}): { postNormEps: number; postNormWeightOffset: boolean };
export function postNormContractMatchesBase(config: {
  rmsNormEps: number;
  rmsNormWeightOffset: boolean;
  postNormEps: number;
  postNormWeightOffset: boolean;
}): boolean;
