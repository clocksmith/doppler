export const SOURCE_ACQUISITION_SCHEMA_ID: 'doppler.source-acquisition/v1';

export declare function createSourceAcquisitionReceipt(
  policy: Record<string, unknown>,
  adapters: {
    listFiles(): Promise<string[]>;
    statFile(file: string): Promise<number>;
    hashFile(file: string): Promise<string>;
  }
): Promise<Record<string, unknown>>;
