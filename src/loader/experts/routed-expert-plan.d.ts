import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { RuntimeModelContract } from '../../inference/runtime-model.js';
import type { TensorLocation } from '../loader-types.js';
export interface RoutedExpertPlane {
  role: string;
  name: string;
  location: TensorLocation;
  bytes: number;
  expertBytes: number;
}
export interface RoutedExpertBankPlan {
  layer: number;
  numExperts: number;
  intermediateSize: number;
  hiddenSize: number;
  planes: RoutedExpertPlane[];
  bankBytes: number;
  expertBytes: number;
}
export declare function planRoutedExpertBank(manifest: RDRRManifest | RuntimeModelContract, locations: Map<string, TensorLocation>, layer: number): RoutedExpertBankPlan;
