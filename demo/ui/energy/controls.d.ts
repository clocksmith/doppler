export function updateEnergyStatus(message: string): void;
export function getEnergyDemoById(id: string): Record<string, unknown> | null;
export function setEnergyMetricLabels(problem: string): void;
export function toggleEnergyProblemControls(problem: string): void;
export function syncEnergyDemoSelection(): void;
export function populateEnergyDemoSelect(): void;
export function applyEnergyDemoDefaults(demo: Record<string, unknown>): void;
