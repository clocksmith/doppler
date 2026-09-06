export interface PackServePolicy {
  schema: 'doppler.pack-serve/v1';
  maxRequestBytes: number;
  maxResponseBytes: number;
  maxOutputBytes: number;
  maxDurationMs: number;
  allowedOrigins: string[];
}
export function normalizePackServePolicy(value: unknown): Readonly<PackServePolicy>;
