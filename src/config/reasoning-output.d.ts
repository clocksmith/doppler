export interface ReasoningOutputContract {
  readonly schema: 'doppler.reasoning-output/v1';
  readonly openingToken: Readonly<{ id: number; text: string }>;
  readonly closingToken: Readonly<{ id: number; text: string }>;
}
export function validateReasoningOutput(value: unknown): Readonly<ReasoningOutputContract> | null | undefined;
