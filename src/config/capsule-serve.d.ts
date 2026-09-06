export interface CapsuleServePolicy {
  schema: 'doppler.capsule-serve/v1';
  maxRequestBytes: number;
  maxResponseBytes: number;
  maxOutputBytes: number;
  maxDurationMs: number;
  allowedOrigins: string[];
}
export function normalizeCapsuleServePolicy(value: unknown): Readonly<CapsuleServePolicy>;
