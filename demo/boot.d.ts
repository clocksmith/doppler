export function boot(): Promise<void>;
export function setBootStatus(text: string, detail?: string): void;
export function stopBootProgress(failed?: boolean): void;
export function updateBootModelProgress(event: { phase?: string; message?: string } | null): void;
