export declare const REGISTERED_VARIANT_CALIBRATION_PLAN_SCHEMA:
  'doppler.registered-variant-calibration-plan/v1';
export declare const REGISTERED_VARIANT_CALIBRATION_RECEIPT_SCHEMA:
  'doppler.registered-variant-calibration-receipt/v1';

export declare function digestRegisteredVariantDescriptor(
  operation: string,
  variantId: string,
  descriptor: Record<string, unknown>
): string;

export declare function validateRegisteredVariantCalibrationPlan(
  plan: Record<string, unknown>,
  registry: Record<string, unknown>
): Record<string, unknown>;

export declare function calibrateRegisteredVariants(
  plan: Record<string, unknown>,
  options: {
    registry: Record<string, unknown>;
    runCorrectness(input: Record<string, unknown>): Promise<Record<string, unknown>>;
    evaluatePerformance(input: Record<string, unknown>): Promise<Record<string, unknown>>;
  }
): Promise<Record<string, unknown>>;
