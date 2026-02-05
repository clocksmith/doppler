export function showProgressOverlay(title?: string): void;
export function hideProgressOverlay(): void;
export function setProgressPhase(phase: string, percent: number, label?: string): void;
export function updateProgressFromLoader(info?: { stage?: string; progress?: number; message?: string } | null): void;
