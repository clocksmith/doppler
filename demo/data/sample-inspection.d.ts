import type { DopplerModelInspectionReceipt } from '../../src/client/model-host/model-session.js';
export declare const SAMPLE_INSPECTION_RECEIPT: Omit<DopplerModelInspectionReceipt, 'activationTrace'> & {
  prompt: string;
  recording: { publishedAt: string; receiptSha256: string; modelId: string };
};
