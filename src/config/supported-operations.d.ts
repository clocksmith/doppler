export const SUPPORTED_OPERATIONS_SCHEMA_ID: 'doppler.supported-operations/v1';

export const SUPPORTED_EXECUTION_V1_OPS: ReadonlySet<string>;
export const SUPPORTED_RUNTIME_OPS: ReadonlySet<string>;

export interface SupportedOpDescriptor {
  readonly section: ReadonlyArray<'preLayer' | 'prefill' | 'decode' | 'postLayer'>;
  readonly description: string;
  readonly kernelFamily: ReadonlyArray<string>;
}

export function getSupportedOpsForFamily(family: string): ReadonlySet<string> | null;
export function listSupportedFamilies(): ReadonlyArray<string>;
export function describeOp(op: string): SupportedOpDescriptor | null;
