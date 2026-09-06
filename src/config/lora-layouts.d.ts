export type LoRAWeightLayoutName = 'input-major' | 'peft';
export interface LoRAWeightLayout {
  readonly name: LoRAWeightLayoutName;
  readonly transposeB: boolean;
  readonly aRankAxis: 0 | 1;
  readonly bRankAxis: 0 | 1;
  readonly identityLayout: 'peft' | null;
}
export function resolveLoRAWeightLayout(name?: LoRAWeightLayoutName): LoRAWeightLayout;
export function resolveLoRAFormatLayout(format: string): LoRAWeightLayout;
