import type { ActivationFrame, ActivationTraceReceipt } from '../../../src/debug/activation-trace.js';
import type { ActivationRecording } from './data.js';
import type { DopplerModelInspectionReceipt } from '../../../src/client/model-host/model-session.js';
export interface GridElements {
  'activation-export': HTMLButtonElement; 'activation-play': HTMLButtonElement;
  'activation-live': HTMLButtonElement; 'activation-canvas': HTMLCanvasElement;
  'activation-record': HTMLInputElement; 'activation-scrub': HTMLInputElement;
  'activation-position': HTMLInputElement; 'activation-scale': HTMLInputElement;
  'activation-speed': HTMLSelectElement;
  'activation-caption': HTMLElement; 'activation-layout': HTMLElement;
  'activation-status': HTMLElement; 'activation-cell': HTMLElement;
  'activation-player': HTMLElement; 'activation-detail-controls': HTMLElement;
}
export declare function isActivationRecordingEnabled(): boolean;
export declare function beginActivationRecording(): void;
export declare function acceptActivationFrame(event: { type: 'activation-frame'; frame: ActivationFrame; trace: ActivationTraceReceipt }): void;
export declare function finishActivationRecording(status: 'complete' | 'stopped' | 'error', receipt?: DopplerModelInspectionReceipt | null): void;
export declare function importActivationRecording(value: unknown): void;
export declare function isActivationRecordingFile(value: unknown): boolean;
export declare function openActivationRecording(value: unknown, byteLength: number): void;
export declare function getActivationRecording(): ActivationRecording | null;
export declare function initActivationGrid(): void;
