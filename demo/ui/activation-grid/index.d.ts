import type { ActivationFrame, ActivationTraceReceipt } from '../../../src/debug/activation-trace.js';
import type { ActivationRecording } from './data.js';
import type { DopplerModelInspectionReceipt } from '../../../src/client/model-host/model-session.js';
export interface GridElements {
  'activity-previews': HTMLElement; 'activity-gallery': HTMLElement; 'activity-selectors': HTMLElement;
  'activity-source': HTMLElement; 'activation-view': HTMLElement;
  'activity-intro': HTMLElement;
  'activity-maps': HTMLElement; 'activity-title': HTMLElement; 'activity-method': HTMLElement;
  'activity-explanation': HTMLElement; 'activity-coverage': HTMLElement;
  'activity-paths': HTMLElement; 'activity-paths-button': HTMLButtonElement;
  'activity-layer-map': HTMLElement;
  'activity-legend': HTMLElement; 'activity-update': HTMLElement; 'activation-timeline': HTMLElement;
  'activity-mode': HTMLSelectElement; 'activity-layer': HTMLSelectElement; 'activity-playback': HTMLSelectElement;
  'activity-weight': HTMLSelectElement; 'activity-detail': HTMLSelectElement;
  'activity-page': HTMLInputElement; 'activity-compare-file': HTMLInputElement;
  'activity-page-control': HTMLElement; 'activity-availability': HTMLElement;
  'activity-weight-controls': HTMLElement; 'activity-compare-controls': HTMLElement;
  'activity-example': HTMLButtonElement; 'activity-compare': HTMLButtonElement; 'activity-adjacent': HTMLButtonElement;
  'activation-export': HTMLButtonElement; 'activation-play': HTMLButtonElement;
  'activation-live': HTMLButtonElement; 'activation-canvas': HTMLCanvasElement;
  'activation-record': HTMLInputElement; 'activation-scrub': HTMLInputElement;
  'activation-record-view': HTMLInputElement; 'activation-record-view-label': HTMLElement;
  'activation-position': HTMLInputElement; 'activation-scale': HTMLInputElement;
  'activation-speed': HTMLSelectElement;
  'activation-caption': HTMLElement; 'activation-layout': HTMLElement;
  'activation-status': HTMLElement; 'activation-cell': HTMLElement;
  'activation-player': HTMLElement; 'activation-detail-controls': HTMLElement;
  'activity-unlisted': HTMLElement; 'activity-hint': HTMLElement; 'activity-unsure': HTMLElement;
  'activity-go': HTMLButtonElement;
  'activity-unsure-prev': HTMLButtonElement; 'activity-unsure-next': HTMLButtonElement;
}
export declare function isActivationRecordingEnabled(): boolean;
/** Shows the "only the open view" choice when the loaded runtime can record a chosen subset. */
export declare function setCaptureSelectionSupport(model: { inspect: { listPolicies?: () => { activationTrace?: { activitySelection?: unknown } }[] } } | null): void;
/** The capture selection for the next answer; undefined records everything the policy allows. */
export declare function activityCaptureRequest(): { roles: string[] } | undefined;
export declare function clearActivationRecording(): void;
export declare function beginActivationRecording(prompt?: string): void;
export declare function acceptActivationFrame(event: { type: 'activation-frame'; frame: ActivationFrame; trace: ActivationTraceReceipt }): void;
export declare function finishActivationRecording(status: 'complete' | 'stopped' | 'error', receipt?: DopplerModelInspectionReceipt | null, outputText?: string | null): void;
export declare function importActivationRecording(value: unknown): void;
export declare function isActivationRecordingFile(value: unknown): boolean;
export declare function openActivationRecording(value: unknown, byteLength: number, options?: { preserveDraft?: boolean }): void;
export declare function getActivationRecording(): ActivationRecording | null;
export declare function initActivationGrid(options?: { onOpenRecording?: (recording: ActivationRecording, options?: { preserveDraft?: boolean }) => void; onPlayback?: (recording: ActivationRecording, point: import('./views.js').PlaybackPoint) => void; unsureWords?: () => { position: number; probability: number }[]; onUnsureWord?: (position: number) => void }): void;

export declare function loadActivityExample(mode?: string, options?: { opening?: boolean }): Promise<void>;
export declare function cancelActivityExample(): void;
