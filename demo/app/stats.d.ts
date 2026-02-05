export function getStatsMode(): string;
export function setStatLabels(labels: Record<string, string>): void;
export function setRunLogLabels(labels: Record<string, string>): void;
export function updatePerformancePanel(snapshot: Record<string, unknown>): void;
export function updateMemoryPanel(snapshot: Record<string, unknown>): void;
export function updateMemoryControls(): void;
export function renderRunLog(): void;
export function recordRunLog(stats: Record<string, unknown>, label?: string, modeOverride?: string): void;
