import type { ChoiceScoringResult } from '../config/choice-scoring.js';

export const CHOICE_CALIBRATION_SCHEMA: 'doppler.choice-calibration/v1';
export const CALIBRATED_CHOICE_RESULT_SCHEMA: 'doppler.calibrated-choice-scoring/v1';

export interface LabeledChoiceScoringExample {
  readonly result: Pick<ChoiceScoringResult, 'choices'>;
  readonly targetId: string;
}

export interface ChoiceCalibrationMetrics {
  readonly sampleCount: number;
  readonly temperature: number;
  readonly accuracy: number;
  readonly meanLogLoss: number;
  readonly brierScore: number;
  readonly ece: number;
}

export interface ChoiceTemperatureCalibration {
  readonly schema: typeof CHOICE_CALIBRATION_SCHEMA;
  readonly method: 'temperature-scaling';
  readonly datasetId: string | null;
  readonly temperature: number;
  readonly before: ChoiceCalibrationMetrics;
  readonly after: ChoiceCalibrationMetrics;
}

export interface CalibratedChoiceEntry {
  readonly id: string;
  readonly label: string;
  readonly tokenId: number;
  readonly logit: number;
  readonly probability: number;
}

export interface CalibratedChoiceScoringResult {
  readonly schema: typeof CALIBRATED_CHOICE_RESULT_SCHEMA;
  readonly sourceSchema: ChoiceScoringResult['schema'];
  readonly calibration: Readonly<{
    schema: typeof CHOICE_CALIBRATION_SCHEMA;
    method: 'temperature-scaling';
    temperature: number;
    datasetId: string | null;
  }>;
  readonly selectedId: string;
  readonly promptTokenCount: number;
  readonly choices: ReadonlyArray<CalibratedChoiceEntry>;
}

export function computeChoiceProbabilities(logits: ReadonlyArray<number>, temperature?: number): number[];

export function evaluateChoiceCalibration(
  examples: ReadonlyArray<LabeledChoiceScoringExample>,
  options?: { temperature?: number; bins?: number }
): ChoiceCalibrationMetrics;

export function fitChoiceTemperatureCalibration(
  examples: ReadonlyArray<LabeledChoiceScoringExample>,
  options?: { minTemperature?: number; maxTemperature?: number; bins?: number; datasetId?: string }
): ChoiceTemperatureCalibration;

export function applyChoiceTemperatureCalibration(
  result: ChoiceScoringResult,
  calibration: Pick<ChoiceTemperatureCalibration, 'schema' | 'method' | 'temperature' | 'datasetId'>
): CalibratedChoiceScoringResult;
