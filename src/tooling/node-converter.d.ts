import type { ConvertResult, ConvertProgress } from '../converter/core.js';
import type { ConverterConfigSchema, ConverterExecutionConfigSchema } from '../config/schema/converter.schema.js';
import type { ConversionReportSchema } from '../config/schema/conversion-report.schema.js';
import type { SavedReportInfo } from '../storage/reports.js';

export interface NodeConvertOptions {
  inputDir: string;
  outputDir?: string;
  modelId?: string;
  configPath?: string;
  converterConfig?: Partial<ConverterConfigSchema>;
  execution?: Partial<ConverterExecutionConfigSchema>;
  onProgress?: (progress: ConvertProgress) => void;
}

export declare function convertSafetensorsDirectory(options: NodeConvertOptions): Promise<
  Omit<ConvertResult, 'totalSize'> & {
    report: ConversionReportSchema;
    reportInfo: SavedReportInfo;
    modelType: string;
    outputDir: string;
  }
>;
